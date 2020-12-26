from typing import Callable, Dict, Optional

import numpy as np
import jax.numpy as jnp
import numpyro as npy
import numpyro.distributions as dist
from jax import nn, random


from covid19.distributions import NegativeBinomial
from covid19.utils import missing_data_plate


class UTLAModel(object):
    """
    UTLA model

    Parameters:
    -----------
    B: spline basis vectors
    C: country indicator variables
    U: UTLA indicator variables
    N: Population
    y: incidences (locations x time)
    """

    def guide(self, B, C, U, N, y):
        num_countries = len(np.unique(C))
        num_timesteps, num_basis = B.shape
        num_locations = N.shape[0]

        num_utla = len(np.unique(U)) - num_countries

        plate_time = npy.plate("time", num_timesteps, dim=-1)
        plate_locations = npy.plate("locations", num_locations, dim=-2)

        # spatial pooling
        μ_β_loc = npy.param("μ_β_loc", jnp.zeros((num_countries, num_basis)))
        μ_σ_scale = npy.param(
            "μ_σ_scale",
            jnp.ones((num_countries, num_basis)),
            constraint=dist.constraints.positive,
        )
        σ_β_scale = npy.param(
            "σ_β_scale",
            jnp.ones((num_countries, num_basis)),
            constraint=dist.constraints.positive,
        )

        μ_β = npy.sample("μ_β", dist.Normal(μ_β_loc, μ_σ_scale))
        σ_β = npy.sample("σ_β", dist.HalfNormal(σ_β_scale))

        μ_utla_loc = npy.param("μ_utla_loc", jnp.zeros((num_utla, num_basis)))
        μ_utla_scale = npy.param(
            "μ_utla_scale",
            jnp.ones((num_utla, num_basis)),
            constraint=dist.constraints.positive,
        )

        μ_utla = npy.sample(
            "μ_utla",
            dist.Normal(
                μ_utla_loc,
                μ_utla_scale,
            ),
        )

        # mean / sd for parameter s
        β_loc = npy.param("β_loc", jnp.zeros((num_locations, num_basis)))
        β_scale = npy.param(
            "β_scale",
            jnp.stack(num_locations * [jnp.eye(num_basis)]),
            constraint=dist.constraints.lower_cholesky,
        )

        # cov = jnp.matmul(β_σ, jnp.transpose(β_σ, (0, 2, 1)))
        β = npy.sample("β", dist.MultivariateNormal(β_loc, scale_tril=β_scale))  # cov

        τ_μ = npy.param("τ_μ", jnp.ones(num_locations).reshape(-1, 1))
        τ_σ = npy.param("τ_σ", jnp.ones(num_locations).reshape(-1, 1))
        τ = npy.sample("τ", dist.Normal(τ_μ, τ_σ))

        with plate_locations:
            with plate_time:
                λ = npy.deterministic(
                    "λ",
                    jnp.exp(jnp.log(N.reshape(-1, 1)) + jnp.inner(β, B)),
                )

    def model(self, B, C, U, N, y):
        mask_context = missing_data_plate(y)
        num_countries = len(np.unique(C))
        num_utla = len(np.unique(U)) - num_countries
        num_timesteps, num_basis = B.shape
        num_locations = N.shape[0]

        plate_time = npy.plate("time", num_timesteps, dim=-1)
        plate_locations = npy.plate("locations", num_locations, dim=-2)

        μ_β = npy.sample(
            "μ_β",
            dist.Normal(
                jnp.zeros((num_countries, num_basis)),
                10 * jnp.ones((num_countries, num_basis)),
            ),
        )
        σ_β = npy.sample("σ_β", dist.HalfNormal(jnp.ones((num_countries, num_basis))))

        μ_utla = npy.sample(
            "μ_utla",
            dist.Normal(
                μ_β[0],
                10 * jnp.ones((num_utla, num_basis)),
            ),
        )
        μ = jnp.concatenate([μ_β, μ_utla])

        β = npy.sample(
            "β",
            dist.MultivariateNormal(
                μ[U],
                σ_β[C].reshape(num_locations, num_basis, 1)
                * jnp.eye(num_basis).reshape(1, num_basis, num_basis),
            ),
        )

        τ = npy.sample(
            "τ",
            dist.Normal(
                jnp.zeros(num_locations).reshape(-1, 1),
                10 * jnp.ones(num_locations).reshape(-1, 1),
            ),
        )

        with plate_locations:
            with plate_time:
                λ = npy.deterministic(
                    "λ",
                    jnp.exp(jnp.log(N.reshape(-1, 1)) + jnp.inner(β, B)),
                )

                with mask_context:
                    npy.sample(
                        "y", NegativeBinomial(λ, jnp.exp(τ)), obs=np.nan_to_num(y)
                    )


class CountryModel(object):
    def guide(self, B, C, N, y):

        num_countries = len(np.unique(C))
        num_timesteps, num_basis = B.shape
        num_locations = N.shape[0]

        plate_time = npy.plate("time", num_timesteps, dim=-1)
        plate_locations = npy.plate("locations", num_locations, dim=-2)

        # spatial pooling
        μ_β_loc = npy.param("μ_β_loc", jnp.zeros((num_countries, num_basis)))
        μ_σ_scale = npy.param(
            "μ_σ_scale",
            jnp.ones((num_countries, num_basis)),
            constraint=dist.constraints.positive,
        )
        σ_β_scale = npy.param(
            "σ_β_scale",
            jnp.ones((num_countries, num_basis)),
            constraint=dist.constraints.positive,
        )

        μ_β = npy.sample("μ_β", dist.Normal(μ_β_loc, μ_σ_scale))
        σ_β = npy.sample("σ_β", dist.HalfNormal(σ_β_scale))

        # mean / sd for parameter s
        β_loc = npy.param("β_loc", jnp.zeros((num_locations, num_basis)))
        β_scale = npy.param(
            "β_scale",
            jnp.stack(num_locations * [jnp.eye(num_basis)]),
            constraint=dist.constraints.lower_cholesky,
        )

        # cov = jnp.matmul(β_σ, jnp.transpose(β_σ, (0, 2, 1)))
        β = npy.sample("β", dist.MultivariateNormal(β_loc, scale_tril=β_scale))  # cov

        τ_μ = npy.param("τ_μ", jnp.ones(num_locations).reshape(-1, 1))
        τ_σ = npy.param("τ_σ", jnp.ones(num_locations).reshape(-1, 1))
        τ = npy.sample("τ", dist.Normal(τ_μ, τ_σ))

        with plate_locations:
            with plate_time:
                λ = npy.deterministic(
                    "λ",
                    jnp.exp(jnp.log(N.reshape(-1, 1)) + jnp.inner(β, B)),
                )

    def model(self, B, C, N, y):
        # handles missing data
        mask_context = missing_data_plate(y)
        # nu
        num_countries = len(np.unique(C))
        num_timesteps, num_basis = B.shape
        num_locations = N.shape[0]

        plate_time = npy.plate("time", num_timesteps, dim=-1)
        plate_locations = npy.plate("locations", num_locations, dim=-2)

        μ_β = npy.sample(
            "μ_β",
            dist.Normal(
                jnp.zeros((num_countries, num_basis)),
                10 * jnp.ones((num_countries, num_basis)),
            ),
        )
        σ_β = npy.sample("σ_β", dist.HalfNormal(jnp.ones((num_countries, num_basis))))

        β = npy.sample(
            "β",
            dist.MultivariateNormal(
                μ_β[C],
                σ_β[C].reshape(num_locations, num_basis, 1)
                * jnp.eye(num_basis).reshape(1, num_basis, num_basis),
            ),
        )

        τ = npy.sample(
            "τ",
            dist.Normal(
                jnp.zeros(num_locations).reshape(-1, 1),
                10 * jnp.ones(num_locations).reshape(-1, 1),
            ),
        )

        with plate_locations:
            with plate_time:
                λ = npy.deterministic(
                    "λ",
                    jnp.exp(jnp.log(N.reshape(-1, 1)) + jnp.inner(β, B)),
                )

                with mask_context:
                    npy.sample(
                        "y", NegativeBinomial(λ, jnp.exp(τ)), obs=np.nan_to_num(y)
                    )


class PoissonExponentialLink(object):

    def model(self, N, c, x, y):
        """
        Pooled linear trend model with Poisson likelihood log/exponetial
        link function.
        """
        n_c = len(np.unique(c))
        n_la = N.shape[0]
        
        mask_context = missing_data_plate(y)
        plate_time = npy.plate("time", x.shape[0], dim=-1)
        plate_local_authorities = npy.plate("authorities", n_la, dim=-2)

        μ_α = npy.sample("μ_α", dist.Normal(jnp.zeros(n_c), 10 * jnp.ones(n_c)))
        σ_α = npy.sample("σ_α", dist.HalfNormal(jnp.ones(n_c)))

        μ_β = npy.sample("μ_β", dist.Normal(jnp.zeros(n_c), 10 * jnp.ones(n_c)))
        σ_β = npy.sample("σ_β", dist.HalfNormal(jnp.ones(n_c)))

        with plate_local_authorities:
            α_offset = npy.sample("α_offset", dist.Normal(jnp.zeros((n_la, 1)), 1))
            α = npy.deterministic(
                "α", μ_α[c].reshape(-1, 1) + α_offset * σ_α[c].reshape(-1, 1)
            )
            # α = npy.sample('α', dist.Delta(α), obs=α)
            β_offset = npy.sample("β_offset", dist.Normal(jnp.zeros((n_la, 1)), 1))
            β = npy.deterministic(
                "β", μ_β[c].reshape(-1, 1) + β_offset * σ_β[c].reshape(-1, 1)
            )
            # β = npy.sample('β', dist.Delta(β), obs=β)

        with plate_time:
            with plate_local_authorities:
                λ = npy.deterministic(
                    "λ",
                    jnp.exp(
                        jnp.log(N.reshape(-1, 1))
                        + α.reshape(-1, 1)
                        + β.reshape(-1, 1) * x.reshape(1, -1)
                    ),
                )
                with mask_context:
                    npy.sample("y", dist.Poisson(λ), obs=np.nan_to_num(y))

    def guide(self, N, c, x, y):
        """
        Guide for the pooled linear trend model with Poisson likelihood log/exponetial
        link function.
        """
        n_c = len(np.unique(c))
        n_la = N.shape[0]

        plate_time = npy.plate("time", x.shape[0], dim=-1)
        plate_local_authorities = npy.plate("authorities", n_la, dim=-2)

        # variational distributions for country intercepts
        μ_α_loc = npy.param("μ_α_loc", jnp.zeros(n_c))
        μ_α_scale = npy.param(
            "μ_α_scale", jnp.ones(n_c), constraint=dist.constraints.positive
        )
        μ_α = npy.sample("μ_α", dist.Normal(loc=μ_α_loc, scale=μ_α_scale))

        σ_α_loc = npy.param("σ_α_loc", jnp.zeros(n_c))
        σ_α_scale = npy.param(
            "σ_α_scale", 0.1 * jnp.ones(n_c), constraint=dist.constraints.positive
        )

        σ_α = npy.sample(
            "σ_α",
            dist.TransformedDistribution(
                dist.Normal(loc=σ_α_loc, scale=σ_α_scale),
                transforms=dist.transforms.ExpTransform(),
            ),
        )
        # print ('sig alpha', σ_α)

        # variational distributions for country slopes
        μ_β_loc = npy.param("μ_β_loc", jnp.zeros(n_c))
        μ_β_scale = npy.param(
            "μ_β_scale", jnp.ones(n_c), constraint=dist.constraints.positive
        )
        μ_β = npy.sample("μ_β", dist.Normal(loc=μ_β_loc, scale=μ_β_scale))

        σ_β_loc = npy.param("σ_β_loc", jnp.zeros(n_c))
        σ_β_scale = npy.param(
            "σ_β_scale", 0.1 * jnp.ones(n_c), constraint=dist.constraints.positive
        )

        σ_β = npy.sample(
            "σ_β",
            dist.TransformedDistribution(
                dist.Normal(loc=σ_β_loc, scale=σ_β_scale),
                transforms=dist.transforms.ExpTransform(),
            ),
        )

        with plate_local_authorities:
            α_offset_loc = npy.param("α_offset_loc", jnp.zeros((n_la, 1)))
            α_offset_scale = npy.param(
                "α_offset_scale",
                0.001 * jnp.ones((n_la, 1)),
                constraint=dist.constraints.positive,
            )
            α_offset = npy.sample(
                "α_offset", dist.Normal(loc=α_offset_loc, scale=α_offset_scale)
            )
            α = npy.deterministic(
                "α", μ_α[c].reshape(-1, 1) + α_offset * σ_α[c].reshape(-1, 1)
            )

            β_offset_loc = npy.param("β_offset_loc", jnp.zeros((n_la, 1)))
            β_offset_scale = npy.param(
                "β_offset_scale",
                0.001 * jnp.ones((n_la, 1)),
                constraint=dist.constraints.positive,
            )
            β_offset = npy.sample(
                "β_offset", dist.Normal(loc=β_offset_loc, scale=β_offset_scale)
            )
            β = npy.deterministic(
                "β", μ_β[c].reshape(-1, 1) + β_offset * σ_β[c].reshape(-1, 1)
            )

        with plate_time:
            with plate_local_authorities:
                λ = npy.deterministic(
                    "λ",
                    N.reshape(-1, 1)
                    * jnp.nan_to_num(
                        jnp.exp(α.reshape(-1, 1) + β.reshape(-1, 1) * x.reshape(1, -1))
                    ),
                )


class PoissonSoftplusLink(object):
    def model(self, N, c, x, y):
        """
        Pooled linear trend model with Poisson likelihood log/softplus
        link function.
        """
        n_c = len(jnp.unique(c))
        n_la = y.shape[0]

        plate_time = npy.plate("time", y.shape[1], dim=-1)
        plate_local_authorities = npy.plate("authorities", y.shape[0], dim=-2)

        μ_α = npy.sample("μ_α", dist.Normal(jnp.zeros(n_c), 10 * jnp.ones(n_c)))
        σ_α = npy.sample("σ_α", dist.HalfNormal(jnp.ones(n_c)))

        μ_β = npy.sample("μ_β", dist.Normal(jnp.zeros(n_c), 10 * jnp.ones(n_c)))
        σ_β = npy.sample("σ_β", dist.HalfNormal(jnp.ones(n_c)))

        with plate_local_authorities:
            α_offset = npy.sample("α_offset", dist.Normal(jnp.zeros((n_la, 1)), 1))
            α = npy.deterministic(
                "α", μ_α[c].reshape(-1, 1) + α_offset * σ_α[c].reshape(-1, 1)
            )
            # α = npy.sample('α', dist.Delta(α), obs=α)
            β_offset = npy.sample("β_offset", dist.Normal(jnp.zeros((n_la, 1)), 1))
            β = npy.deterministic(
                "β", μ_β[c].reshape(-1, 1) + β_offset * σ_β[c].reshape(-1, 1)
            )
            # β = npy.sample('β', dist.Delta(β), obs=β)

        with plate_time:
            with plate_local_authorities:
                λ = npy.deterministic(
                    "λ",
                    N.reshape(-1, 1)
                    * nn.softplus(
                        α.reshape(-1, 1) + β.reshape(-1, 1) * x.reshape(1, -1)
                    ),
                )
                npy.sample("y", dist.Poisson(λ), obs=y)

    def guide(self, N, c, x, y):
        n_c = len(np.unique(c))
        n_la = y.shape[0]

        plate_time = npy.plate("time", y.shape[1], dim=-1)
        plate_local_authorities = npy.plate("authorities", y.shape[0], dim=-2)

        # variational distributions for country intercepts
        μ_α_loc = npy.param("μ_α_loc", jnp.zeros(n_c))
        μ_α_scale = npy.param(
            "μ_α_scale", jnp.ones(n_c), constraint=dist.constraints.positive
        )
        μ_α = npy.sample("μ_α", dist.Normal(loc=μ_α_loc, scale=μ_α_scale))

        σ_α_loc = npy.param("σ_α_loc", jnp.zeros(n_c))
        σ_α_scale = npy.param(
            "σ_α_scale", 0.1 * jnp.ones(n_c), constraint=dist.constraints.positive
        )

        σ_α = npy.sample(
            "σ_α",
            dist.TransformedDistribution(
                dist.Normal(loc=σ_α_loc, scale=σ_α_scale),
                transforms=dist.transforms.ExpTransform(),
            ),
        )
        # print ('sig alpha', σ_α)

        # variational distributions for country slopes
        μ_β_loc = npy.param("μ_β_loc", jnp.zeros(n_c))
        μ_β_scale = npy.param(
            "μ_β_scale", jnp.ones(n_c), constraint=dist.constraints.positive
        )
        μ_β = npy.sample("μ_β", dist.Normal(loc=μ_β_loc, scale=μ_β_scale))

        σ_β_loc = npy.param("σ_β_loc", jnp.zeros(n_c))
        σ_β_scale = npy.param(
            "σ_β_scale", 0.1 * jnp.ones(n_c), constraint=dist.constraints.positive
        )

        σ_β = npy.sample(
            "σ_β",
            dist.TransformedDistribution(
                dist.Normal(loc=σ_β_loc, scale=σ_β_scale),
                transforms=dist.transforms.ExpTransform(),
            ),
        )

        with plate_local_authorities:
            α_offset_loc = npy.param("α_offset_loc", jnp.zeros((n_la, 1)))
            α_offset_scale = npy.param(
                "α_offset_scale",
                0.1 * jnp.ones((n_la, 1)),
                constraint=dist.constraints.positive,
            )
            α_offset = npy.sample(
                "α_offset", dist.Normal(loc=α_offset_loc, scale=α_offset_scale)
            )
            α = npy.deterministic(
                "α", μ_α[c].reshape(-1, 1) + α_offset * σ_α[c].reshape(-1, 1)
            )

            β_offset_loc = npy.param("β_offset_loc", jnp.zeros((n_la, 1)))
            β_offset_scale = npy.param(
                "β_offset_scale",
                0.1 * jnp.ones((n_la, 1)),
                constraint=dist.constraints.positive,
            )
            β_offset = npy.sample(
                "β_offset", dist.Normal(loc=β_offset_loc, scale=β_offset_scale)
            )
            β = npy.deterministic(
                "β", μ_β[c].reshape(-1, 1) + β_offset * σ_β[c].reshape(-1, 1)
            )

        with plate_time:
            with plate_local_authorities:
                μ = nn.softplus(
                    jnp.log(N.reshape(-1, 1))
                    + α.reshape(-1, 1)
                    + β.reshape(-1, 1) * x.reshape(1, -1)
                )
                λ = npy.deterministic("λ", μ)


class NegativeBinomialSoftplusLink(object):
    def model(N, c, x, y):
        n_c = len(np.unique(c))
        n_la = y.shape[0]
        n_t = y.shape[1]

        plate_time = npy.plate("time", y.shape[1], dim=-1)
        plate_local_authorities = npy.plate("authorities", y.shape[0], dim=-2)

        μ_α = npy.sample("μ_α", dist.Normal(jnp.zeros(n_c), 10 * jnp.ones(n_c)))
        σ_α = npy.sample("σ_α", dist.HalfNormal(jnp.ones(n_c)))

        μ_β = npy.sample("μ_β", dist.Normal(jnp.zeros(n_c), 10 * jnp.ones(n_c)))
        σ_β = npy.sample("σ_β", dist.HalfNormal(jnp.ones(n_c)))

        with plate_local_authorities:
            α_offset = npy.sample("α_offset", dist.Normal(jnp.zeros((n_la, 1)), 1))
            α = npy.deterministic(
                "α", μ_α[c].reshape(-1, 1) + α_offset * σ_α[c].reshape(-1, 1)
            )
            # α = npy.sample('α', dist.Delta(α), obs=α)
            β_offset = npy.sample("β_offset", dist.Normal(jnp.zeros((n_la, 1)), 1))
            β = npy.deterministic(
                "β", μ_β[c].reshape(-1, 1) + β_offset * σ_β[c].reshape(-1, 1)
            )
            # β = npy.sample('β', dist.Delta(β), obs=β)

            # with plate_time:
            τ = npy.sample("τ", dist.HalfNormal(jnp.ones((n_la, 1))))

        with plate_time:
            with plate_local_authorities:
                λ = npy.deterministic(
                    "λ",
                    N.reshape(-1, 1)
                    * nn.softplus(
                        α.reshape(-1, 1) + β.reshape(-1, 1) * x.reshape(1, -1)
                    ),
                )
                npy.sample("y", dist.GammaPoisson(λ, τ), obs=y)

    def guide(N, c, x, y):
        n_c = len(np.unique(c))
        n_la = y.shape[0]
        n_t = y.shape[1]

        plate_time = npy.plate("time", y.shape[1], dim=-1)
        plate_local_authorities = npy.plate("authorities", y.shape[0], dim=-2)

        # variational distributions for country intercepts
        μ_α_loc = npy.param("μ_α_loc", jnp.zeros(n_c))
        μ_α_scale = npy.param(
            "μ_α_scale", jnp.ones(n_c), constraint=dist.constraints.positive
        )
        μ_α = npy.sample("μ_α", dist.Normal(loc=μ_α_loc, scale=μ_α_scale))

        σ_α_loc = npy.param("σ_α_loc", jnp.zeros(n_c))
        σ_α_scale = npy.param(
            "σ_α_scale", 0.1 * jnp.ones(n_c), constraint=dist.constraints.positive
        )

        σ_α = npy.sample(
            "σ_α",
            dist.TransformedDistribution(
                dist.Normal(loc=σ_α_loc, scale=σ_α_scale),
                transforms=dist.transforms.ExpTransform(),
            ),
        )
        # print ('sig alpha', σ_α)

        # variational distributions for country slopes
        μ_β_loc = npy.param("μ_β_loc", jnp.zeros(n_c))
        μ_β_scale = npy.param(
            "μ_β_scale", jnp.ones(n_c), constraint=dist.constraints.positive
        )
        μ_β = npy.sample("μ_β", dist.Normal(loc=μ_β_loc, scale=μ_β_scale))

        σ_β_loc = npy.param("σ_β_loc", jnp.zeros(n_c))
        σ_β_scale = npy.param(
            "σ_β_scale", 0.1 * jnp.ones(n_c), constraint=dist.constraints.positive
        )

        σ_β = npy.sample(
            "σ_β",
            dist.TransformedDistribution(
                dist.Normal(loc=σ_β_loc, scale=σ_β_scale),
                transforms=dist.transforms.ExpTransform(),
            ),
        )

        with plate_local_authorities:
            α_offset_loc = npy.param("α_offset_loc", jnp.zeros((n_la, 1)))
            α_offset_scale = npy.param(
                "α_offset_scale",
                0.1 * jnp.ones((n_la, 1)),
                constraint=dist.constraints.positive,
            )
            α_offset = npy.sample(
                "α_offset", dist.Normal(loc=α_offset_loc, scale=α_offset_scale)
            )
            α = npy.deterministic(
                "α", μ_α[c].reshape(-1, 1) + α_offset * σ_α[c].reshape(-1, 1)
            )

            β_offset_loc = npy.param("β_offset_loc", jnp.zeros((n_la, 1)))
            β_offset_scale = npy.param(
                "β_offset_scale",
                0.1 * jnp.ones((n_la, 1)),
                constraint=dist.constraints.positive,
            )
            β_offset = npy.sample(
                "β_offset", dist.Normal(loc=β_offset_loc, scale=β_offset_scale)
            )
            β = npy.deterministic(
                "β", μ_β[c].reshape(-1, 1) + β_offset * σ_β[c].reshape(-1, 1)
            )

            # with plate_time:
            τ_loc = npy.param("τ_loc", jnp.zeros((n_la, 1)))
            τ_scale = npy.param(
                "τ_scale",
                0.1 * jnp.ones((n_la, 1)),
                constraint=dist.constraints.positive,
            )
            τ = npy.sample(
                "τ",
                dist.TransformedDistribution(
                    dist.Normal(loc=τ_loc, scale=τ_scale),
                    transforms=dist.transforms.ExpTransform(),
                ),
            )

        with plate_time:
            with plate_local_authorities:
                μ = nn.softplus(
                    jnp.log(N.reshape(-1, 1))
                    + α.reshape(-1, 1)
                    + β.reshape(-1, 1) * x.reshape(1, -1)
                )
                λ = npy.deterministic("λ", μ)
