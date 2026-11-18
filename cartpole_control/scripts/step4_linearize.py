"""Step 4: linearize at the upright equilibrium.

Pass: analytic A, B match finite differences; exactly one positive real eigenvalue;
controllability matrix has rank 4.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # import modules from cartpole_control/

import numpy as np  # noqa: E402
from scipy.linalg import expm  # noqa: E402

from dynamics import f  # noqa: E402
from linearize import (X_UP, controllability_matrix, linearize_analytic,  # noqa: E402
                       linearize_fd, linearize_mujoco_step)
from sim_utils import load, report  # noqa: E402


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)
    A, B = linearize_analytic()
    A_fd, B_fd = linearize_fd(f, X_UP, 0.0)
    print("A =\n", A, "\nB =\n", B.ravel())

    err = max(np.abs(A - A_fd).max(), np.abs(B - B_fd).max())
    report("analytic A, B vs finite differences", err < 1e-6, f"max err {err:.2e}")

    eig = np.linalg.eigvals(A)
    n_pos = int(np.sum((eig.real > 1e-9) & (np.abs(eig.imag) < 1e-9)))
    report("exactly one positive real eigenvalue", n_pos == 1, f"eig(A) = {np.round(eig, 4)}")

    rank = np.linalg.matrix_rank(controllability_matrix(A, B))
    report("controllability rank 4", rank == 4, f"rank = {rank}")

    # Discrete A, B from MuJoCo vs exact discretization of our continuous model
    model, data = load()
    dt = model.opt.timestep
    # mjd_transitionFD does not support RK4, so difference mj_step directly
    Ad, Bd = linearize_mujoco_step(model, data, X_UP, 0.0)
    Phi = expm(np.block([[A, B], [np.zeros((1, 5))]]) * dt)   # zero-order-hold discretization
    errd = max(np.abs(Ad - Phi[:4, :4]).max(), np.abs(Bd - Phi[:4, 4:]).max())
    report("MuJoCo step Ad, Bd vs expm(A dt)", errd < 1e-6, f"max err {errd:.2e}")


if __name__ == "__main__":
    main()
