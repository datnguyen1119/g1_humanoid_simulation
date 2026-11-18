"""Watch a controller live in the MuJoCo viewer.

Usage (from cartpole_control/ or the project root):
    python scripts/demo.py --mode full      # start at the bottom: swing-up, then LQR catches it
    python scripts/demo.py --mode lqr       # start at pi - 0.2, LQR balances
    python scripts/demo.py --mode swingup   # swing-up only, never switches to LQR

In the viewer: double-click the pole, then Ctrl + right-drag to push it and watch the controller react.
"""
import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # import modules from cartpole_control/

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402

from controllers import (SWINGUP_GAINS, HybridController, design_lqr, lqr_control,  # noqa: E402
                         swingup_control)
from sim_utils import OUT_DIR, get_state, load, set_state  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "lqr", "swingup"], default="full")
    args = ap.parse_args()

    model, data = load()
    K, S = design_lqr()
    if args.mode == "lqr":
        controller, x0 = (lambda x: lqr_control(x, K)), np.array([0.0, np.pi - 0.2, 0.0, 0.0])
    elif args.mode == "swingup":
        controller, x0 = (lambda x: swingup_control(x, **SWINGUP_GAINS)), np.array([0.0, 0.05, 0.0, 0.0])
    else:
        npz = OUT_DIR / "step5_lqr.npz"
        rho = float(np.load(npz)["rho"]) if npz.exists() else 29.0
        controller, x0 = HybridController(K, S, rho, SWINGUP_GAINS), np.array([0.0, 0.05, 0.0, 0.0])

    set_state(model, data, x0)
    last_mode = None
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t0 = time.time()
            data.ctrl[0] = controller(get_state(data))
            mode = getattr(controller, "mode", args.mode)
            if mode != last_mode:
                print(f"t={data.time:5.2f}s  mode: {mode}")
                last_mode = mode
            mujoco.mj_step(model, data)
            viewer.sync()
            dt = model.opt.timestep - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
