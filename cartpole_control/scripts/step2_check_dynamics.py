"""Step 2: compare hand-derived M, C, g with MuJoCo.

Pass: max error < 1e-8 over 100 random states, for M (vs mj_fullM) and C qd + g (vs qfrc_bias).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # import modules from cartpole_control/

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from dynamics import coriolis, f, gravity, mass_matrix  # noqa: E402
from sim_utils import load, report, set_state  # noqa: E402


def main() -> None:
    model, data = load()
    rng = np.random.default_rng(0)
    M_mj = np.zeros((model.nv, model.nv))
    err_M = err_bias = err_qacc = 0.0

    for _ in range(100):
        x = np.array([rng.uniform(-1.5, 1.5), rng.uniform(-np.pi, np.pi),
                      rng.uniform(-3, 3), rng.uniform(-10, 10)])
        u = rng.uniform(-20, 20)
        set_state(model, data, x)
        data.ctrl[0] = u
        mujoco.mj_forward(model, data)   # recompute at exactly this state, do not step

        q, qd = x[:2], x[2:]
        mujoco.mj_fullM(model, data, M_mj)  # MuJoCo >= 3.x: (model, data, dst)
        err_M = max(err_M, np.abs(mass_matrix(q) - M_mj).max())
        err_bias = max(err_bias, np.abs(coriolis(q, qd) @ qd + gravity(q) - data.qfrc_bias).max())
        err_qacc = max(err_qacc, np.abs(f(x, u)[2:] - data.qacc).max())

    report("M vs mj_fullM", err_M < 1e-8, f"max err {err_M:.2e}")
    report("C qd + g vs qfrc_bias", err_bias < 1e-8, f"max err {err_bias:.2e}")
    report("qdd from f(x, u) vs qacc", err_qacc < 1e-8, f"max err {err_qacc:.2e}")


if __name__ == "__main__":
    main()
