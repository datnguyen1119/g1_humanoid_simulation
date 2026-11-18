"""Linearization of the cart-pole around an equilibrium (step 4).

    d/dt x_err = A x_err + B u_err,   x_err = x - x0,  u_err = u - u0
Upright equilibrium: x0 = [0, pi, 0, 0], u0 = 0.

At an equilibrium qd = 0, so the C(q, qd) qd term is quadratic in qd and drops out:
    qdd = M0^-1 (B u - dg/dq * dq)
    A = [[0, I], [-M0^-1 dg/dq, 0]],   B = [[0], [M0^-1 B]]
"""
import numpy as np

from dynamics import B as B_FORCE, mass_matrix
from params import P, CartPoleParams

X_UP = np.array([0.0, np.pi, 0.0, 0.0])


def linearize_analytic(p: CartPoleParams = P) -> tuple[np.ndarray, np.ndarray]:
    """A (4, 4), B (4, 1) at the upright equilibrium, derived by hand."""
    M0 = mass_matrix(X_UP[:2], p)                     # cos(pi) = -1
    dg_dq = np.array([[0.0, 0.0],
                      [0.0, p.m_p * p.g * p.l * np.cos(np.pi)]])  # d/dth (m_p g l sin th)
    A = np.zeros((4, 4))
    A[:2, 2:] = np.eye(2)
    A[2:, :2] = -np.linalg.solve(M0, dg_dq)
    B = np.zeros((4, 1))
    B[2:, 0] = np.linalg.solve(M0, B_FORCE)
    return A, B


def linearize_fd(f, x0: np.ndarray, u0: float, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """A, B by central finite differences of f(x, u)."""
    n = len(x0)
    A = np.zeros((n, n))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        A[:, i] = (f(x0 + dx, u0) - f(x0 - dx, u0)) / (2 * eps)
    B = ((f(x0, u0 + eps) - f(x0, u0 - eps)) / (2 * eps)).reshape(n, 1)
    return A, B


def controllability_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """[B, AB, A^2 B, A^3 B], shape (4, 4)."""
    cols = [B]
    for _ in range(A.shape[0] - 1):
        cols.append(A @ cols[-1])
    return np.hstack(cols)


def linearize_mujoco_step(model, data, x0: np.ndarray, u0: float,
                          eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Discrete Ad, Bd of one MuJoCo step (x_next = Ad x + Bd u) by central differences.

    Replaces mujoco.mjd_transitionFD, which does not support the RK4 integrator.
    """
    import mujoco

    def step(x, u):
        mujoco.mj_resetData(model, data)
        data.qpos[:], data.qvel[:] = x[:2], x[2:]
        data.ctrl[0] = u
        mujoco.mj_step(model, data)
        return np.concatenate([data.qpos, data.qvel])

    n = len(x0)
    Ad = np.zeros((n, n))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        Ad[:, i] = (step(x0 + dx, u0) - step(x0 - dx, u0)) / (2 * eps)
    Bd = ((step(x0, u0 + eps) - step(x0, u0 - eps)) / (2 * eps)).reshape(n, 1)
    return Ad, Bd
