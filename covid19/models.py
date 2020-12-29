from typing import Callable, Dict, Optional

import numpy as np
import jax.numpy as jnp
import numpyro as npy
import numpyro.distributions as dist
from jax import nn, random
from jax.ops import index, index_update


from covid19.distributions import NegativeBinomial
from covid19.utils import missing_data_plate


class RelaxedStrainModel(object):
    RHO = "ρ"

    MU_BETA = "μ_β"
    MU_UTLA = "μ_utla"
    SIGMA_BETA = "σ_β"
    BETA_BASE = "β_base"

    A0 = "a0"
    B0 = "b0"
    A = "a"
    B = "b"
    C = "c"

    BETA_STRAIN = "β_strain"
    B_STRAIN = "b_strain"

    MU_NEW = "μ_new"
    MU_BASE = "μ_base"
    MU_STRAIN = "μ_strain"
    MU = "μ"

    LAMBDA_STRAIN = "λ_strain"
    LAMBDA_BASE = "λ_base"
    LAMBDA = "λ"

    R_STRAIN = "R_strain"
    R_BASE = "R_base"

    P = "p"
    SPECIMEN = "specimen"
    STRAIN = "strain"

    LOC = "_loc"
    SCALE = "_scale"

    def model(self, B, Bdiff, X, S, C, U, N, y, strain, total):
        mask_context = missing_data_plate(y)

        num_strain_loc = S.shape[0]
        num_strain_time = X.shape[0]

        num_countries = len(np.unique(C))
        num_utla = len(np.unique(U)) - num_countries
        num_timesteps, num_basis = B.shape
        num_locations = N.shape[0]

        plate_time = npy.plate("time", num_timesteps, dim=-1)
        plate_locations = npy.plate("locations", num_locations, dim=-2)

        plate_strain_time = npy.plate("strain_time", num_strain_time, dim=-1)
        plate_strain_locations = npy.plate("strain_locations", num_strain_loc, dim=-2)

        # dispersion parameter for lads
        ρ = npy.sample(
            self.RHO,
            dist.Normal(
                jnp.zeros(num_locations).reshape(-1, 1),
                10 * jnp.ones(num_locations).reshape(-1, 1),
            ),
        )

        μ_β = npy.sample(
            self.MU_BETA,
            dist.Normal(
                jnp.zeros((num_countries, num_basis)),
                10 * jnp.ones((num_countries, num_basis)),
            ),
        )
        σ_β = npy.sample(
            self.SIGMA_BETA, dist.HalfNormal(jnp.ones((num_countries, num_basis)))
        )

        μ_utla = npy.sample(
            self.MU_UTLA,
            dist.Normal(
                μ_β[0],
                10 * jnp.ones((num_utla, num_basis)),
            ),
        )
        μ = jnp.concatenate([μ_β, μ_utla])

        # beta base for all of the 382 local authorities
        # draws from the UTLA/Country priors (indexed with U and C)
        β_base = npy.sample(
            self.BETA_BASE,
            dist.MultivariateNormal(
                μ[U],
                σ_β[C].reshape(num_locations, num_basis, 1)
                * jnp.eye(num_basis).reshape(1, num_basis, num_basis),
            ),
        )

        # draw scaling factor a, S indexes the locations for which strain information is
        # available, dim(β_new) = 194 x num_basis
        # a = jnp.array([1.])#n
        # ab0 = npy.sample('ab0', dist.MultivariateNormal(jnp.zeros([2]), jnp.eye(2)))
        a0 = npy.sample(self.A0, dist.Normal(jnp.ones([1]), jnp.ones([1])))
        b0 = npy.sample(self.B0, dist.Normal(jnp.ones([1]), jnp.ones([1])))
        a = jnp.exp(
            npy.sample(
                self.A,
                dist.Normal(jnp.ones(num_strain_loc) * a0, jnp.ones(num_strain_loc)),
            )
            * 0.05
        )
        b = (
            npy.sample(
                self.B,
                dist.Normal(jnp.ones(num_strain_loc) * b0, jnp.ones(num_strain_loc)),
            )
            * 0.05
        )
        c = (
            npy.sample(
                self.C, dist.Normal(jnp.zeros(num_strain_loc), jnp.ones(num_strain_loc))
            )
            - 5
        )

        β_new = a.reshape(-1, 1) * β_base[S]
        # β strain has the full dimension 384 x num_basis, zeros at the locations where strain
        # information is not available and elsewhere β_new
        β_strain = jnp.zeros((num_locations, num_basis))
        β_strain = npy.deterministic(
            self.BETA_STRAIN, index_update(β_strain, index[S, :], β_new)
        )

        b_strain = jnp.zeros((num_locations))
        b_strain = npy.deterministic(self.B_STRAIN, index_update(b_strain, S, b))

        μ_new = npy.deterministic(
            self.MU_NEW,
            jnp.exp(
                jnp.inner(β_new, B)
                + b.reshape(-1, 1) * (jnp.arange(num_timesteps).reshape(1, -1))
                + c.reshape(-1, 1)
            ),
        )
        # print(μ_new)
        # print(b.reshape(-1, 1) * (jnp.arange(num_timesteps).reshape(1,-1)))

        μ_base = npy.deterministic(self.MU_BASE, jnp.exp(jnp.inner(β_base, B)))
        # μ_strain contains at locations for which strain information is available
        # μ_new, else 0
        μ_strain = jnp.zeros((num_locations, num_timesteps))
        μ_strain = npy.deterministic(
            self.MU_STRAIN, index_update(μ_strain, index[S, :], μ_new)
        )

        μ = npy.deterministic(
            self.MU,
            μ_strain + μ_base,
        )

        λ_strain = npy.deterministic(self.LAMBDA_STRAIN, N.reshape(-1, 1) * μ_strain)
        λ_base = npy.deterministic(self.LAMBDA_BASE, N.reshape(-1, 1) * μ_base)
        λ = npy.deterministic(self.LAMBDA, N.reshape(-1, 1) * μ)

        R_strain = npy.deterministic(
            self.R_STRAIN,
            jnp.exp((jnp.inner(β_strain, Bdiff) + b_strain.reshape(-1, 1)) * 6.5),
        )
        R_base = npy.deterministic(self.R_BASE, jnp.exp(jnp.inner(β_base, Bdiff) * 6.5))

        with plate_locations:
            with plate_time:
                with mask_context:
                    npy.sample(
                        self.SPECIMEN,
                        NegativeBinomial(λ, jnp.exp(ρ)),
                        obs=np.nan_to_num(y),
                    )

        p = npy.deterministic(self.P, μ_strain / μ)

        with plate_strain_locations:
            with plate_strain_time:
                npy.sample(
                    self.STRAIN,
                    dist.Binomial(probs=p[:, X][S, :], total_count=total),
                    obs=strain,
                )

    def guide(self, B, Bdiff, X, S, C, U, N, y, strain, total):

        num_strain_loc = S.shape[0]
        num_strain_time = X.shape[0]

        num_countries = len(np.unique(C))
        num_timesteps, num_basis = B.shape
        num_locations = N.shape[0]

        num_utla = len(np.unique(U)) - num_countries

        # spatial pooling
        μ_β_loc = npy.param(
            self.MU_BETA + self.LOC, jnp.zeros((num_countries, num_basis))
        )
        μ_β_scale = npy.param(
            self.MU_BETA + self.SCALE,
            jnp.ones((num_countries, num_basis)),
            constraint=dist.constraints.positive,
        )
        σ_β_scale = npy.param(
            self.SIGMA_BETA + self.SCALE,
            jnp.ones((num_countries, num_basis)),
            constraint=dist.constraints.positive,
        )

        μ_β = npy.sample(self.MU_BETA, dist.Normal(μ_β_loc, μ_β_scale))
        σ_β = npy.sample(self.SIGMA_BETA, dist.HalfNormal(σ_β_scale))

        μ_utla_loc = npy.param(
            self.MU_UTLA + self.LOC, jnp.zeros((num_utla, num_basis))
        )
        μ_utla_scale = npy.param(
            self.MU_UTLA + self.SCALE,
            jnp.ones((num_utla, num_basis)),
            constraint=dist.constraints.positive,
        )

        μ_utla = npy.sample(
            self.MU_UTLA,
            dist.Normal(
                μ_utla_loc,
                μ_utla_scale,
            ),
        )

        # mean / sd for parameter s
        β_loc = npy.param(
            self.BETA_BASE + self.LOC, jnp.zeros((num_locations, num_basis))
        )
        β_scale = npy.param(
            self.BETA_BASE + self.SCALE,
            jnp.stack(num_locations * [jnp.eye(num_basis)]),
            constraint=dist.constraints.lower_cholesky,
        )

        # cov = jnp.matmul(β_σ, jnp.transpose(β_σ, (0, 2, 1)))
        β_base = npy.sample(
            self.BETA_BASE, dist.MultivariateNormal(β_loc, scale_tril=β_scale)
        )  # cov

        a0_loc = npy.param(self.A0 + self.LOC, jnp.zeros(1))
        a0_scale = npy.param(
            self.A0 + self.SCALE, jnp.ones(1), constraint=dist.constraints.positive
        )
        b0_loc = npy.param(self.B0 + self.LOC, jnp.zeros(1))
        b0_scale = npy.param(
            self.B0 + self.SCALE, jnp.ones(1), constraint=dist.constraints.positive
        )

        a0 = npy.sample(self.A0, dist.Normal(a0_loc, a0_scale))
        b0 = npy.sample(self.B0, dist.Normal(b0_loc, b0_scale))
        # ab0_scale *= jnp.array([[1.,0],[.1,1.]])
        # ab0 = npy.sample('ab0', dist.MultivariateNormal(ab0_loc, scale_tril=ab0_scale))
        # ab0 = npy.sample('ab0', dist.Normal(ab0_loc, scale=ab0_scale))

        a_loc = npy.param(self.A + self.LOC, jnp.zeros(num_strain_loc))
        a_scale = npy.param(
            self.A + self.SCALE,
            jnp.ones(num_strain_loc),
            constraint=dist.constraints.positive,
        )
        b_loc = npy.param(self.B + self.LOC, jnp.zeros(num_strain_loc))
        b_scale = npy.param(
            self.B + self.SCALE,
            jnp.ones(num_strain_loc),
            constraint=dist.constraints.positive,
        )
        c_loc = npy.param(self.C + self.LOC, jnp.zeros(num_strain_loc))
        c_scale = npy.param(
            self.C + self.SCALE,
            jnp.ones(num_strain_loc),
            constraint=dist.constraints.positive,
        )

        a = npy.sample(self.A, dist.Normal(a_loc, a_scale))
        b = npy.sample(self.B, dist.Normal(b_loc, b_scale))
        c = npy.sample(self.C, dist.Normal(c_loc, c_scale))

        ρ_loc = npy.param(self.RHO + self.LOC, jnp.ones(num_locations).reshape(-1, 1))
        ρ_scale = npy.param(
            self.RHO + self.SCALE, jnp.ones(num_locations).reshape(-1, 1)
        )
        ρ = npy.sample(self.RHO, dist.Normal(ρ_loc, ρ_scale))
