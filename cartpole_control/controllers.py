"""Controllers for the cart-pole: LQR (step 5), energy swing-up (step 6), hybrid switch (step 7).

All controllers take the full state x = [x, theta, xd, thetad] and return the cart force u (N),
already clipped to +-P.u_max, so the force you log is the force MuJoCo applies.
"""
import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are

from dynamics import pole_energy
from params import P, CartPoleParams


def wrap_angle(a: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def upright_error(x: np.ndarray) -> np.ndarray:
    """x_err = [x, wrap(theta - pi), xd, thetad]."""
    return np.array([x[0], wrap_angle(x[1] - np.pi), x[2], x[3]])


def lqr_continuous(A, B, Q, R) -> tuple[np.ndarray, np.ndarray]:
    """Return gain K (1, 4) and Riccati solution S (4, 4): u = -K x, K = R^-1 B^T S."""
    S = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ S)
    return K, S


def lqr_discrete(Ad, Bd, Q, R) -> tuple[np.ndarray, np.ndarray]:
    """Discrete-time LQR: K = (R + B^T S B)^-1 B^T S A."""
    S = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ S @ Bd, Bd.T @ S @ Ad)
    return K, S


def lqr_control(x: np.ndarray, K: np.ndarray, p: CartPoleParams = P) -> float:
    """u = -K x_err, clipped."""
    u = float(-(K @ upright_error(x))[0])
    return float(np.clip(u, -p.u_max, p.u_max))


def pfl_force(x: np.ndarray, xdd_des: float, p: CartPoleParams = P) -> float:
    """Collocated partial feedback linearization: the force u that makes xdd = xdd_des.

    Pole row:  J thdd + m_p l c xdd + m_p g l s = 0   ->  thdd = -(m_p l c xdd + m_p g l s) / J
    Cart row:  (m_c + m_p) xdd + m_p l c thdd - m_p l s thd^2 = u
    Substitute thdd into the cart row.
    """
    s, c, thd = np.sin(x[1]), np.cos(x[1]), x[3]
    J = p.I + p.m_p * p.l**2
    mt = p.m_c + p.m_p
    mlc = p.m_p * p.l * c
    return (mt - mlc**2 / J) * xdd_des - mlc * p.m_p * p.g * p.l * s / J - p.m_p * p.l * s * thd**2


def swingup_control(x: np.ndarray, k_E: float, k_x: float, k_xd: float,
                    p: CartPoleParams = P) -> float:
    """Energy shaping + PD on cart position, mapped to force by collocated PFL.

    E_pole' = -m_p l cos(th) thd xdd, so xdd = k_E * E_err * thd * cos(th) gives
    E_pole' = -m_p l k_E cos^2(th) thd^2 E_err  ->  E_err -> 0 for k_E > 0.
    """
    E_err = pole_energy(x, p) - p.m_p * p.g * p.l
    xdd_des = k_E * E_err * x[3] * np.cos(x[1]) - k_x * x[0] - k_xd * x[2]
    return float(np.clip(pfl_force(x, xdd_des, p), -p.u_max, p.u_max))


class HybridController:
    """Swing-up until x_err^T S x_err < rho_in, then LQR.

    Hysteresis: return to swing-up only if the cost grows above rho_out > rho_in,
    so the mode does not chatter at the boundary.
    """

    def __init__(self, K: np.ndarray, S: np.ndarray, rho: float, swingup_gains: dict,
                 rho_out_factor: float = 3.0):
        self.K, self.S = K, S
        self.rho_in, self.rho_out = rho, rho * rho_out_factor
        self.gains = swingup_gains
        self.mode = "swingup"

    def cost(self, x: np.ndarray) -> float:
        e = upright_error(x)
        return float(e @ self.S @ e)

    def __call__(self, x: np.ndarray) -> float:
        V = self.cost(x)
        if self.mode == "swingup" and V < self.rho_in:
            self.mode = "lqr"
        elif self.mode == "lqr" and V > self.rho_out:
            self.mode = "swingup"
        if self.mode == "lqr":
            return lqr_control(x, self.K)
        return swingup_control(x, **self.gains)


# LQR weights chosen in step 5: from theta = pi +- 0.2, |u| peaks at ~13 N and settles in ~2.2 s
LQR_Q = np.diag([10.0, 50.0, 1.0, 1.0])   # [x, theta_err, xd, thetad]
LQR_R = np.array([[0.1]])


def design_lqr() -> tuple[np.ndarray, np.ndarray]:
    """K, S of the continuous LQR at the upright equilibrium with LQR_Q, LQR_R."""
    from linearize import linearize_analytic
    A, B = linearize_analytic()
    return lqr_continuous(A, B, LQR_Q, LQR_R)

# Swing-up gains chosen in step 6: from theta = 0.05, cart stays within ~0.9 m, |u| < 17 N
SWINGUP_GAINS = {"k_E": 5.0, "k_x": 4.0, "k_xd": 3.0}
