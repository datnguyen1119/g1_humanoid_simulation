"""Physical parameters of the cart-pole. Must match model/cartpole.xml."""
from dataclasses import dataclass
import pathlib

MODEL_PATH = pathlib.Path(__file__).resolve().parent / "model" / "cartpole.xml"


@dataclass(frozen=True)
class CartPoleParams:
    m_c: float = 1.0      # cart mass (kg)
    m_p: float = 0.2      # pole mass (kg), uniform rod
    L: float = 0.6        # pole length (m)
    l: float = 0.3        # pivot -> pole COM (m)
    I: float = 0.006      # pole inertia about its COM (kg m^2) = m_p L^2 / 12
    g: float = 9.81       # gravity (m/s^2)
    u_max: float = 20.0   # actuator limit (N)
    x_max: float = 2.0    # rail limit (m)


P = CartPoleParams()
