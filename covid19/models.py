from typing import Callable, Dict, Optional

import numpy as np
import jax.numpy as jnp
import numpyro as npy
import numpyro.distributions as dist
from jax import nn, random


from covid19.distributions import NegativeBinomial
from covid19.utils import missing_data_plate


class RelaxedStrainModel(object):
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
    
    def model(self, B, Bdiff, X, S, C, U, N, y, strain, total):
        mask_context = c19.missing_data_plate(y)
        
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
        τ = npy.sample(
            "τ",
            dist.Normal(
                jnp.zeros(num_locations).reshape(-1, 1),
                10 * jnp.ones(num_locations).reshape(-1, 1),
            ),
        )
        
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
        
        # beta base for all of the 382 local authorities
        # draws from the UTLA/Country priors (indexed with U and C)
        β_base = npy.sample(
            "β_base",
            dist.MultivariateNormal(
                μ[U],
                σ_β[C].reshape(num_locations, num_basis, 1)
                * jnp.eye(num_basis).reshape(1, num_basis, num_basis),
            ),
        )
        
        # draw scaling factor a, S indexes the locations for which strain information is 
        # available, dim(β_new) = 194 x num_basis
        #a = jnp.array([1.])#n
        #ab0 = npy.sample('ab0', dist.MultivariateNormal(jnp.zeros([2]), jnp.eye(2)))
        a0 =  npy.sample('a0', dist.Normal(jnp.ones([1]), jnp.ones([1])))
        b0 =  npy.sample('b0', dist.Normal(jnp.ones([1]), jnp.ones([1])))
        a = jnp.exp(npy.sample('a', dist.Normal(jnp.ones(num_strain_loc) * a0, jnp.ones(num_strain_loc))) * 0.05)
        b = npy.sample('b', dist.Normal(jnp.ones(num_strain_loc) * b0, jnp.ones(num_strain_loc))) * 0.05
        c = npy.sample('c', dist.Normal(jnp.zeros(num_strain_loc), jnp.ones(num_strain_loc))) - 5
        
        β_new = a.reshape(-1, 1) * β_base[S]
        # β strain has the full dimension 384 x num_basis, zeros at the locations where strain
        # information is not available and elsewhere β_new
        β_strain = jnp.zeros((num_locations, num_basis))
        β_strain = npy.deterministic('β_strain', index_update(β_strain, index[S, :], β_new))        
        
        b_strain = jnp.zeros((num_locations))
        b_strain = npy.deterministic('b_strain', index_update(b_strain, S, b))        
        

        μ_new = npy.deterministic(
             'μ_new', 
                    jnp.exp(jnp.inner(β_new, B) + b.reshape(-1, 1) * (jnp.arange(num_timesteps).reshape(1,-1)) + c.reshape(-1,1) )
        )
                #print(μ_new)
                #print(b.reshape(-1, 1) * (jnp.arange(num_timesteps).reshape(1,-1)))
                
        μ_base = npy.deterministic('μ_base', jnp.exp(jnp.inner(β_base, B)))
                #μ_strain contains at locations for which strain information is available
                # μ_new, else 0
        μ_strain = jnp.zeros((num_locations, num_timesteps))
        μ_strain = npy.deterministic('μ_strain', index_update(μ_strain, index[S, :], μ_new))
            
        μ = npy.deterministic(
                    "μ",
                    μ_strain + μ_base,
                )

        
        λ_strain = npy.deterministic('λ_strain', N.reshape(-1,1) * μ_strain)
        λ_base = npy.deterministic('λ_base', N.reshape(-1,1) * μ_base)        
        λ = npy.deterministic('λ', N.reshape(-1,1) * μ)
                
        R_strain = npy.deterministic('R_strain', jnp.exp((jnp.inner(β_strain, Bdiff) + b_strain.reshape(-1, 1)) * 6.5))
        R_base = npy.deterministic('R_base', jnp.exp(jnp.inner(β_base, Bdiff) * 6.5))
        R = npy.deterministic('R', jnp.log(R_strain/R_base))

        
        with plate_locations:
            with plate_time:                
                with mask_context:
                    npy.sample(
                        "y", c19.NegativeBinomial(λ, jnp.exp(τ)), obs=np.nan_to_num(y)
                    )
        #print(μ_strain.shape)
        
        p = npy.deterministic('p', μ_strain/μ)
        
        with plate_strain_locations:
            with plate_strain_time:
                #p = npy.deterministic('p', 1 / (1 + jnp.exp(jnp.inner(β_base-β_strain , B[X,:]) -  b.reshape(-1, 1) * (jnp.arange(num_timesteps)[X].reshape(1,-1)) + c.reshape(-1,1))))
                #p = npy.deterministic('p', (μ_strain/μ)[:,X])
                #print(p.shape)
                npy.sample('strain', dist.Binomial(probs=p[:,X][S,:], total_count=total), obs=strain)

    def guide(self, B, Bdiff, X, S, C, U, N, y, strain, total):
        
        num_strain_loc = S.shape[0]
        num_strain_time = X.shape[0]
        
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
        β_base = npy.sample("β_base", dist.MultivariateNormal(β_loc, scale_tril=β_scale))  # cov
        
        #ab0_loc = npy.param('ab0_loc', jnp.zeros(2))
        #ab0_scale = npy.param('ab0_scale', jnp.eye(2), constraint=dist.constraints.lower_cholesky) 
        #ab0_scale = npy.param('ab0_scale', jnp.ones(2), constraint=dist.constraints.positive)
        a0_loc = npy.param('a0_loc', jnp.zeros(1))
        a0_scale = npy.param('a0_scale', jnp.ones(1), constraint=dist.constraints.positive)
        b0_loc = npy.param('b0_loc', jnp.zeros(1))
        b0_scale = npy.param('b0_scale', jnp.ones(1), constraint=dist.constraints.positive)
        
        a0 = npy.sample('a0', dist.Normal(a0_loc, a0_scale))
        b0 = npy.sample('b0', dist.Normal(b0_loc, b0_scale))
        #ab0_scale *= jnp.array([[1.,0],[.1,1.]])
        #ab0 = npy.sample('ab0', dist.MultivariateNormal(ab0_loc, scale_tril=ab0_scale))
        #ab0 = npy.sample('ab0', dist.Normal(ab0_loc, scale=ab0_scale))


        a_loc = npy.param('a_loc', jnp.zeros(num_strain_loc))
        a_scale = npy.param('a_scale', jnp.ones(num_strain_loc), constraint=dist.constraints.positive)
        b_loc = npy.param('b_loc', jnp.zeros(num_strain_loc))
        b_scale = npy.param('b_scale', jnp.ones(num_strain_loc), constraint=dist.constraints.positive)
        c_loc = npy.param('c_loc', jnp.zeros(num_strain_loc))
        c_scale = npy.param('c_scale', jnp.ones(num_strain_loc), constraint=dist.constraints.positive)
        
        a = npy.sample('a', dist.Normal(a_loc, a_scale))
        b = npy.sample('b', dist.Normal(b_loc, b_scale))
        c = npy.sample('c', dist.Normal(c_loc, c_scale))
        
        τ_μ = npy.param("τ_μ", jnp.ones(num_locations).reshape(-1, 1))
        τ_σ = npy.param("τ_σ", jnp.ones(num_locations).reshape(-1, 1))
        τ = npy.sample("τ", dist.Normal(τ_μ, τ_σ))

