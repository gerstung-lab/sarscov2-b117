from contextlib import suppress

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import lax, random


from numpyro import optim
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO

import numpyro as npy
from functools import lru_cache
from scipy.interpolate import BSpline
import scipy.stats as stats

from covid19.types import Guide, Model
import dill

def compute_R(y, mu=14.9, sigma=3.9, a=1, b=5, τ=7):
    
    t_start = np.arange(1, y.shape[0]-τ+1)
    t_end = np.arange(τ, y.shape[0])
    
    p = np.array([discretise_epistm(i, mu, sigma) for i in range(y.shape[0])])
    expectation = (np.arange(0, p.shape[0]) * p).sum()
    
    λ = np.zeros(p.shape[0])
    
    λ[0] = np.nan
    for t in range(1, y.shape[0]):
        λ[t] = np.sum(p[np.arange(t+1)] * y[np.arange(t, -1, -1)])
    print(np.array([ y[t_start[t]:t_end[t]].sum() if t_end[t]+1 > expectation else np.nan for t in range(t_start.shape[0]) ]))    
    a_p = np.array([ a + y[t_start[t]:t_end[t]].sum() if t_end[t]+1 > expectation else np.nan for t in range(t_start.shape[0]) ])
    b_p = np.array([ 1/ (1/b + λ[t_start[t]:t_end[t]].sum()) if t_end[t]+1 > expectation else np.nan for t in range(t_start.shape[0]) ])
    
    return a_p, b_p, λ

def reparametrise_gamma(mean, cv):
    alpha  = 1/cv**2
    beta = mean * cv **2
    return alpha, beta

def discretise_epistm(k, mu, sigma):
    """
    Discretises a gamma distribution according to Cori et al.
    """
    a = ((mu - 1) / sigma) ** 2
    b = sigma ** 2 / (mu - 1)
    
    cdf_gamma = stats.gamma(a=a, scale=b).cdf
    cdf_gamma2 = stats.gamma(a=a+1, scale=b).cdf
    
    res = k * cdf_gamma(k) + (k-2) * cdf_gamma(k-2) - 2 * (k-1) * cdf_gamma(k-1)
    res = res + a * b * (2 * cdf_gamma2(k-1) - cdf_gamma2(k-2) - cdf_gamma2(k))
    
    return max(res, 0)


def create_prediction_table(model, start_date="09/01/2020"):
    data = model.mean("λ", "posterior_predictive")
    pred = pd.DataFrame(np.array(data)).rename(
        columns=lambda x: dict(
            zip(
                range(data.shape[1]),
                [
                    str(day)[:10]
                    for day in pd.date_range(
                        start=start_date, periods=data.shape[1]
                    ).tolist()
                ],
            )
        )[x]
    )
    return pred


def create_spline_basis(
    x, knot_list=None, num_knots=None, degree=3, add_intercept=True
):
    assert ((knot_list is None) and (num_knots is not None)) or (
        (knot_list is not None) and (num_knots is None)
    ), "Define knot_list OR num_knot"
    if knot_list is None:
        knot_list = jnp.quantile(x, q=jnp.linspace(0, 1, num=num_knots))
    else:
        num_knots = len(knot_list)

    knots = jnp.pad(knot_list, (degree, degree), mode="edge")
    B0 = BSpline(knots, jnp.identity(num_knots + 2), k=degree)
    B = B0(x)
    Bdiff = B0.derivative()(x)
    if add_intercept:
        B = jnp.hstack([jnp.ones(B.shape[0]).reshape(-1, 1), B])
        Bdiff = jnp.hstack([jnp.zeros(B.shape[0]).reshape(-1, 1), Bdiff])
    return knot_list, B, Bdiff


def missing_data_plate(y):
    """
    Returns a plate which indicates missing data.
    """
    if y is not None:
        mask = ~np.isnan(y)
        mask_context = npy.handlers.mask(mask=mask)
    else:
        mask_context = suppress()

    return mask_context


class Handler(object):
    def __init__(self):
        pass

    def _select(self, which):
        assert which in [
            "prior",
            "posterior",
            "posterior_predictive",
        ], "Please select from 'prior', 'posterior' or 'posterior_predictive'."
        assert hasattr(self, which), f"NutsHandler did not compute the {which} yet."
        return getattr(self, which)

    def mean(self, param, which="posterior"):
        return self._select(which)[param].mean(0)

    @lru_cache(maxsize=128)
    def hpdi(self, param, which="posterior", *args, **kwargs):
        return hpdi(self._select(which)[param], *args, **kwargs)

    @lru_cache(maxsize=128)
    def quantiles(self, param, which="posterior", *args, **kwargs):
        return jnp.quantile(self._select(which)[param], jnp.array([0.05, 0.95]), axis=0)

    def qlower(self, param, which="posterior", *args, **kwargs):
        return self.quantiles(param, which=which, *args, **kwargs)[0]

    def qupper(self, param, which="posterior", *args, **kwargs):
        return self.quantiles(param, which=which, *args, **kwargs)[1]

    def lower(self, param, which="posterior", *args, **kwargs):
        return self.hpdi(param, which=which, *args, **kwargs)[0]

    def upper(self, param, which="posterior", *args, **kwargs):
        return self.hpdi(param, which=which, *args, **kwargs)[1]

    def ci(self, param, which="posterior"):
        return np.abs(self.mean(param, which) - self.hpdi(param, which))


class SVIHandler(Handler):
    def __init__(
        self,
        model: Model,
        guide: Guide,
        loss: Trace_ELBO = Trace_ELBO(num_particles=1),
        optimizer: optim.optimizers.optimizer = optim.Adam,
        lr: float = 0.001,
        rng_key: int = 254,
        num_epochs: int = 100000,
        num_samples: int = 5000,
        log_func=print,
        log_freq=0,
    ):
        self.model = model
        self.guide = guide
        self.loss = loss
        self.optimizer = optimizer(step_size=lr)
        self.rng_key = random.PRNGKey(rng_key)

        self.svi = SVI(self.model, self.guide, self.optimizer, loss=self.loss)
        self.init_state = None

        self.log_func = log_func
        self.log_freq = log_freq
        self.num_epochs = num_epochs
        self.num_samples = num_samples

        self.loss = None

    def _log(self, epoch, loss, n_digits=4):
        msg = f"epoch: {str(epoch).rjust(n_digits)} loss: {loss: 16.4f}"
        self.log_func(msg)

    def _fit(self, epochs, *args):
        return lax.scan(
            lambda state, i: self.svi.update(state, *args),
            self.init_state,
            jnp.arange(epochs),
        )

    def _update_state(self, state, loss):
        self.state = state
        self.init_state = state
        self.loss = loss if self.loss is None else jnp.concatenate([self.loss, loss])

    def fit(self, *args, **kwargs):
        num_epochs = kwargs.get("num_epochs", self.num_epochs)
        log_freq = kwargs.get("log_freq", self.log_freq)

        if self.init_state is None:
            self.init_state = self.svi.init(self.rng_key, *args)

        if log_freq <= 0:
            state, loss = self._fit(num_epochs, *args)
            self._update_state(state, loss)
        else:
            steps, rest = num_epochs // log_freq, num_epochs % log_freq

            for step in range(steps):
                state, loss = self._fit(log_freq, *args)
                self._log(log_freq * (step + 1), loss[-1])
                self._update_state(state, loss)

            if rest > 0:
                state, loss = self._fit(rest, *args)
                self._update_state(state, loss)

        self.params = self.svi.get_params(state)
        predictive = Predictive(
            self.model,
            guide=self.guide,
            params=self.params,
            num_samples=self.num_samples,
        )
        self.posterior = predictive(self.rng_key, *args)

    def get_posterior_predictive(self, *args, **kwargs):
        """kwargs -> Predictive, args -> predictive"""
        num_samples = kwargs.pop("num_samples", self.num_samples)

        predictive = Predictive(
            self.model,
            guide=self.guide,
            params=self.params,
            num_samples=num_samples,
            **kwargs,
        )
        self.posterior_predictive = predictive(self.rng_key, *args)


class NutsHandler(Handler):
    def __init__(
        self,
        model,
        posterior=None,
        num_warmup=2000,
        num_samples=10000,
        num_chains=1,
        key=0,
        *args,
        **kwargs,
    ):
        self.model = model
        self.rng_key, self.rng_key_ = random.split(random.PRNGKey(key))

        if posterior is not None:
            self.mcmc = posterior
            self.posterior = self.mcmc.get_samples()
        else:
            self.kernel = NUTS(model, **kwargs)
            self.mcmc = MCMC(
                self.kernel, num_warmup, num_samples, num_chains=num_chains
            )

    def _select(self, which):
        assert which in [
            "prior",
            "posterior",
            "posterior_predictive",
        ], "Please select from 'prior', 'posterior' or 'posterior_predictive'."
        assert hasattr(self, which), f"NutsHandler did not compute the {which} yet."
        return getattr(self, which)

    def get_prior(self, *args, **kwargs):
        predictive = Predictive(self.model, num_samples=self.mcmc.num_samples)
        self.prior = predictive(self.rng_key_, *args, **kwargs)

    def get_posterior_predictive(self, *args, **kwargs):
        predictive = Predictive(self.model, self.posterior, **kwargs)
        self.posterior_predictive = predictive(self.rng_key_, *args)

    def fit(self, *args, **kwargs):
        self.mcmc.run(self.rng_key_, *args, **kwargs)
        self.posterior = self.mcmc.get_samples()

    def summary(self, *args, **kwargs):
        self.mcmc.print_summary(*args, **kwargs)

    def dump(self, path):
        with open(path, "wb") as f:
            dill.dump(self.mcmc, f)

    @staticmethod
    def from_dump(model, path):
        with open(path, "rb") as f:
            posterior = dill.load(f)
        return NutsHandler(model, posterior=posterior)


# class SVIHandler(object):
#     def __init__(
#         self,
#         model: Model,
#         guide: Guide,
#         rng_key: int = 0,
#         *,
#         loss: Trace_ELBO = Trace_ELBO(num_particles=1),
#         optim_builder: optim.optimizers.optimizer = optim.Adam,
#     ):
#         """Handling the model and guide for training and prediction
#         Args:
#             model: function holding the numpyro model
#             guide: function holding the numpyro guide
#             rng_key: random key as int
#             loss: loss to optimize
#             optim_builder: builder for an optimizer
#         """
#         self.model = model
#         self.guide = guide
#         self.rng_key = random.PRNGKey(rng_key)  # current random key
#         self.loss = loss
#         self.optim_builder = optim_builder
#         self.svi = None
#         self.svi_state = None
#         self.optim = None
#         self.log_func = print  # overwrite e.g. logger.info(...)

#     def reset_svi(self):
#         """Reset the current SVI state"""
#         self.svi = None
#         self.svi_state = None
#         return self

#     def init_svi(self, X: DeviceArray, *, lr: float, **kwargs):
#         """Initialize the SVI state
#         Args:
#             X: input data
#             lr: learning rate
#             kwargs: other keyword arguments for optimizer
#         """
#         self.optim = self.optim_builder(lr, **kwargs)
#         self.svi = SVI(self.model, self.guide, self.optim, self.loss)
#         svi_state = self.svi.init(self.rng_key, X)
#         if self.svi_state is None:
#             self.svi_state = svi_state
#         return self

#     @property
#     def optim_state(self) -> OptimizerState:
#         """Current optimizer state"""
#         assert self.svi_state is not None, "'init_svi' needs to be called first"
#         return self.svi_state.optim_state

#     @optim_state.setter
#     def optim_state(self, state: OptimizerState):
#         """Set current optimizer state"""
#         self.svi_state = SVIState(state, self.rng_key)

#     def dump_optim_state(self, fh: IO):
#         """Pickle and dump optimizer state to file handle"""
#         pickle.dump(optim.optimizers.unpack_optimizer_state(self.optim_state[1]), fh)
#         return self

#     def load_optim_state(self, fh: IO):
#         """Read and unpickle optimizer state from file handle"""
#         state = optim.optimizers.pack_optimizer_state(pickle.load(fh))
#         iter0 = jnp.array(0)
#         self.optim_state = (iter0, state)
#         return self

#     @property
#     def optim_total_steps(self) -> int:
#         """Returns the number of performed iterations in total"""
#         return int(self.optim_state[0])

#     def _fit(self, X: DeviceArray, n_epochs) -> float:
#         @jit
#         def train_epochs(svi_state, n_epochs):
#             def train_one_epoch(_, val):
#                 loss, svi_state = val
#                 svi_state, loss = self.svi.update(svi_state, X)
#                 return loss, svi_state

#             return lax.fori_loop(0, n_epochs, train_one_epoch, (0.0, svi_state))

#         loss, self.svi_state = train_epochs(self.svi_state, n_epochs)
#         return float(loss / X.shape[0])

#     def _log(self, n_digits, epoch, loss):
#         msg = f"epoch: {str(epoch).rjust(n_digits)} loss: {loss: 16.4f}"
#         self.log_func(msg)

#     def fit(
#         self, X: DeviceArray, *, n_epochs: int, log_freq: int = 0, lr: float, **kwargs
#     ) -> float:
#         """Train but log with a given frequency
#         Args:
#             X: input data
#             n_epochs: total number of epochs
#             log_freq: log loss every log_freq number of eppochs
#             lr: learning rate
#             kwargs: parameters of `init_svi`
#         Returns:
#             final loss of last epoch
#         """
#         self.init_svi(X, lr=lr, **kwargs)
#         if log_freq <= 0:
#             self._fit(X, n_epochs)
#         else:
#             loss = self.svi.evaluate(self.svi_state, X) / X.shape[0]

#             curr_epoch = 0
#             n_digits = len(str(abs(n_epochs)))
#             self._log(n_digits, curr_epoch, loss)

#             for i in range(n_epochs // log_freq):
#                 curr_epoch += log_freq
#                 loss = self._fit(X, log_freq)
#                 self._log(n_digits, curr_epoch, loss)

#             rest = n_epochs % log_freq
#             if rest > 0:
#                 curr_epoch += rest

#                 loss = self._fit(X, rest)
#                 self._log(n_digits, curr_epoch, loss)

#         loss = self.svi.evaluate(self.svi_state, X) / X.shape[0]
#         self.rng_key = self.svi_state.rng_key
#         return float(loss)

#     @property
#     def model_params(self) -> Optional[Dict[str, DeviceArray]]:
#         """Gets model parameters
#         Returns:
#             dict of model parameters
#         """
#         if self.svi is not None:
#             return self.svi.get_params(self.svi_state)
#         else:
#             return None

#     def predict(self, X: DeviceArray, **kwargs) -> DeviceArray:
#         """Predict the parameters of a model specified by `return_sites`
#         Args:
#             X: input data
#             kwargs: keyword arguments for numpro `Predictive`
#         Returns:
#             samples for all sample sites
#         """
#         self.init_svi(X, lr=0.0)  # dummy initialization
#         predictive = Predictive(
#             self.model, guide=self.guide, params=self.model_params, **kwargs
#         )
#         samples = predictive(self.rng_key, X)
#         return samples
