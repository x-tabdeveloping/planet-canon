import jax
import jax.numpy as jnp
from diffrax import Dopri5, ODETerm, diffeqsolve


def eqm(state_vector, t, m1, m2):
    n_dims = state_vector.shape[0] // 4
    x1 = state_vector[:n_dims]
    x2 = state_vector[n_dims : n_dims * 2]
    velocities = state_vector[n_dims * 2 :]
    diff = x2 - x1
    diff_mag = jnp.linalg.norm(diff)
    acc1 = m2 * (x2 - x1) / jnp.power(diff_mag, 3)
    acc2 = m1 * (x1 - x2) / jnp.power(diff_mag, 3)
    return jnp.concatenate((velocities, acc1, acc2))


def make_solver(eqm, solver=Dopri5()):
    term = ODETerm(eqm)

    def solve_eq(init_state_vector, times, m1, m2):
        solution = diffeqsolve(
            term, solver, t0=0, t1=jnp.max(times) + 1, dt0=None, y0=init_state_vector
        )
        return solution.ys

    return solve_eq


def make_position_calculator(solver=Dopri5()):
    solve_eq = make_solver(eqm, solver)

    def get_body_positions(m1, m2, x0, x1, v1, v2, times):
        init_state_vector = jnp.concatenate((x0, x1, v1, v2))
        n_dims = init_state_vector.shape[0] // 4
        ys = solve_eq(init_state_vector, times, m1, m2)
        new_x0 = ys[:, :n_dims]
        new_x1 = ys[:, n_dims : n_dims * 2]
        return new_x0, new_x1

    return get_body_positions
