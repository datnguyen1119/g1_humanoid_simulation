"""Cart-pole equations of motion, derived by hand with Lagrange (step 2).

    M(q) qdd + C(q, qd) qd + g(q) = B u

Conventions (same as model/cartpole.xml):
    q  = [x, theta]          theta = 0 hanging down, pi upright
    qd = [xd, thetad]
    x  = [x, theta, xd, thetad]  (full state, shape (4,))
    pole COM at (x + l sin(theta), -l cos(theta)) relative to the rail

Derivation:
    pole COM velocity  = (xd + l cos(th) thd,  l sin(th) thd)
    T = 1/2 (m_c + m_p) xd^2 + m_p l cos(th) xd thd + 1/2 (I + m_p l^2) thd^2
    V = -m_p g l cos(th)                     (V = 0 at pivot height)
    => M = [[m_c + m_p,        m_p l cos(th)],
            [m_p l cos(th),    I + m_p l^2  ]]
       d/dt(M[0,1] thd) produces -m_p l sin(th) thd^2 in the x equation
       => C = [[0, -m_p l sin(th) thd],
               [0,  0                ]]
       g = dV/dq = [0, m_p g l sin(th)]
"""
import numpy as np

from params import P, CartPoleParams

B = np.array([1.0, 0.0])  # force acts on the cart coordinate only


def mass_matrix(q: np.ndarray, p: CartPoleParams = P) -> np.ndarray:
    """M(q), shape (2, 2). Compare with mujoco.mj_fullM."""
    c = np.cos(q[1])
    return np.array([
        [p.m_c + p.m_p, p.m_p * p.l * c],
        [p.m_p * p.l * c, p.I + p.m_p * p.l**2],
    ])


def coriolis(q: np.ndarray, qd: np.ndarray, p: CartPoleParams = P) -> np.ndarray:
    """C(q, qd), shape (2, 2), such that C @ qd is the Coriolis/centrifugal term."""
    s = np.sin(q[1])
    return np.array([
        [0.0, -p.m_p * p.l * s * qd[1]],
        [0.0, 0.0],
    ])


def gravity(q: np.ndarray, p: CartPoleParams = P) -> np.ndarray:
    """g(q), shape (2,). Note: data.qfrc_bias in MuJoCo equals C @ qd + g."""
    return np.array([0.0, p.m_p * p.g * p.l * np.sin(q[1])])


def f(x: np.ndarray, u: float, p: CartPoleParams = P) -> np.ndarray:
    """State derivative xdot = [qd, qdd], shape (4,). Solve for qdd, do not invert M."""
    q, qd = x[:2], x[2:]
    rhs = B * u - coriolis(q, qd, p) @ qd - gravity(q, p)
    qdd = np.linalg.solve(mass_matrix(q, p), rhs)
    return np.concatenate([qd, qdd])


def energy(x: np.ndarray, p: CartPoleParams = P) -> float:
    """Total energy T + V, with V = 0 at the pivot height."""
    q, qd = x[:2], x[2:]
    T = 0.5 * qd @ mass_matrix(q, p) @ qd
    V = -p.m_p * p.g * p.l * np.cos(q[1])
    return float(T + V)


def pole_energy(x: np.ndarray, p: CartPoleParams = P) -> float:
    """Energy of the pole alone as if the cart were fixed: 1/2 J thd^2 - m_p g l cos(th).

    Used by the swing-up law (Tedrake ch. 3). Equals m_p g l when resting upright.
    """
    J = p.I + p.m_p * p.l**2
    return float(0.5 * J * x[3] ** 2 - p.m_p * p.g * p.l * np.cos(x[1]))
