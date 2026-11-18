"""Hand-written integrators for xdot = f(x, u) (step 3).

Each step function advances the state by one time step dt with u held constant.
"""
from typing import Callable

import numpy as np

Dynamics = Callable[[np.ndarray, float], np.ndarray]


def euler_step(f: Dynamics, x: np.ndarray, u: float, dt: float) -> np.ndarray:
    """Explicit Euler: one evaluation of f. Global error O(dt)."""
    return x + dt * f(x, u)


def rk4_step(f: Dynamics, x: np.ndarray, u: float, dt: float) -> np.ndarray:
    """Classic 4th-order Runge-Kutta: four evaluations of f. Global error O(dt^4)."""
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(step, f: Dynamics, x0: np.ndarray, u: float, dt: float, T: float) -> np.ndarray:
    """Roll out from x0 for T seconds. Return the trajectory, shape (N + 1, 4)."""
    n = int(round(T / dt))
    traj = np.empty((n + 1, len(x0)))
    traj[0] = x0
    for k in range(n):
        traj[k + 1] = step(f, traj[k], u, dt)
    return traj
