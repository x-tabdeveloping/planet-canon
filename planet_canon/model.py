import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal

from planet_canon.two_body_equations import make_position_calculator

get_body_positions = make_position_calculator()


@jax.jit
def loglikelihood_fn(
    observed_positions_0,
    observed_positions_1,
    observed_times_0,
    observed_times_1,
    initial_position_0,
    initial_position_1,
    initial_velocity_0,
    initial_velocity_1,
    covariance_0,
    covariance_1,
    m0,
    m1,
):
    mu_pos_0, mu_pos_1 = get_body_positions(
        m0,
        m1,
        initial_position_0,
        initial_position_1,
        initial_velocity_0,
        initial_velocity_1,
        times=jnp.concat(observed_times_0, observed_times_1),
    )
    loglikelihood_0 = multivariate_normal.logpdf(
        observed_positions_0, mean=mu_pos_0[observed_times_0], cov=covariance_0
    )
    loglikelihood_1 = multivariate_normal.logpdf(
        observed_positions_1, mean=mu_pos_1[observed_times_1], cov=covariance_1
    )
    return jnp.sum(loglikelihood_0) + jnp.sum(loglikelihood_1)
