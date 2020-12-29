import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from jax import lax, random
from numpyro.distributions import constraints
from numpyro.distributions.util import lazy_property, promote_shapes, validate_sample
from jax.scipy.special import betaln, gammaln


class NegativeBinomial(dist.Distribution):
    r"""
    Compound distribution comprising of a gamma-poisson pair, also referred to as
    a gamma-poisson mixture. The ``rate`` parameter for the
    :class:`~numpyro.distributions.Poisson` distribution is unknown and randomly
    drawn from a :class:`~numpyro.distributions.Gamma` distribution.

    :param numpy.ndarray concentration: shape parameter (alpha) of the Gamma distribution.
    :param numpy.ndarray rate: rate parameter (beta) for the Gamma distribution.
    """
    arg_constraints = {
        "concentration": constraints.positive,
        "rate": constraints.positive,
    }
    support = constraints.nonnegative_integer
    is_discrete = True

    def __init__(self, mu, tau, validate_args=None):
        self.mu, self.tau = promote_shapes(mu, tau)
        # converts mean var parametrisation to r and p
        self.r = tau
        self.var = mu + 1 / self.r * mu ** 2
        self.p = (self.var - mu) / self.var
        self._gamma = dist.Gamma(self.r, (1 - self.p) / self.p)
        super(NegativeBinomial, self).__init__(
            self._gamma.batch_shape, validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        key_gamma, key_poisson = random.split(key)
        rate = self._gamma.sample(key_gamma, sample_shape)
        return dist.Poisson(rate).sample(key_poisson)

    @validate_sample
    def log_prob(self, value):

        return (
            self.tau * jnp.log(self.tau)
            - gammaln(self.tau)
            + gammaln(value + self.tau)
            + value * jnp.log(self.mu)
            - jnp.log(self.mu + self.tau) * (self.tau + value)
            - gammaln(value + 1)
        )