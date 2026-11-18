"""Run the ID-QP balance controller of g1_demo.py live in the MuJoCo viewer.

g1_demo.py itself is headless (it simulates, then writes out/log.npz and a video).
This script executes only its setup part (model, helpers, solve_idqp) without modifying it,
then runs its own loop with the passive viewer.

Usage (from humanoid_g1_control/):
    .venv/bin/python sim/run_g1_idqp.py            # scripted scenario, repeats every 10 s
    .venv/bin/python sim/run_g1_idqp.py --once     # play the scenario once, then keep balancing

In the viewer: double-click the torso, then Ctrl + right-drag to push it yourself.
Note: one QP takes ~3-4 ms while the sim step is 2 ms, so playback is slower than real time.
"""
import argparse
import os
import pathlib
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEMO = ROOT / "g1_demo.py"
MARKER = "# ---------------------------------------------------------------- simulation loop"


def load_demo() -> dict:
    """Execute g1_demo.py up to the simulation loop and return its namespace."""
    os.environ.setdefault("G1_XML", str(ROOT / "third_party/mujoco_menagerie/unitree_g1/scene.xml"))
    os.environ["RECORD"] = "0"
    src = DEMO.read_text()
    if MARKER not in src:
        sys.exit(f"cannot find the simulation-loop marker in {DEMO}")
    ns = {"__file__": str(DEMO), "__name__": "g1_demo_setup"}
    exec(compile(src.split(MARKER)[0], str(DEMO), "exec"), ns)
    return ns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="do not restart the scenario")
    args = ap.parse_args()

    g = load_demo()
    m, d, q_nom, TORSO, T_END = g["m"], g["d"], g["q_nom"], g["TORSO"], g["T_END"]
    nv, nu = g["nv"], g["nu"]

    def reset():
        mujoco.mj_resetData(m, d)
        d.qpos[:] = q_nom
        mujoco.mj_forward(m, d)

    reset()
    last_print, last_sync = -1.0, 0.0
    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.5, 140, -12
        while viewer.is_running():
            t_wall = time.perf_counter()
            # after the scenario ends: restart, or (--once) keep balancing; com_ref is constant after 6 s
            if not args.once and d.time >= T_END:
                reset(); print("--- restart scenario ---")
                continue

            x, status, c, cr, _ = g["solve_idqp"](d.time)
            if x is None or not np.all(np.isfinite(x)):
                print(f"QP failed at t={d.time:.3f}: {status}"); break
            d.ctrl[:] = x[nv:nv + nu]

            F = g["push_force"](d.time)
            if F.any():                     # scripted push; otherwise leave xfrc_applied to the viewer
                d.xfrc_applied[TORSO, :3] = F
            elif np.any(d.xfrc_applied[TORSO, :3]) and g["push_force"](d.time - m.opt.timestep).any():
                d.xfrc_applied[TORSO, :3] = 0   # end of a scripted push

            mujoco.mj_step(m, d)
            if time.perf_counter() - last_sync >= 1 / 60:   # render at ~60 Hz, not every 2 ms step
                viewer.sync(); last_sync = time.perf_counter()

            if d.time - last_print >= 0.5:
                last_print = d.time
                tag = "  PUSH" if F.any() else ""
                print(f"t={d.time:5.2f}s  com_err={np.linalg.norm(c - cr) * 1e3:5.1f} mm  {status}{tag}")
            if d.qpos[2] < 0.4:
                print(f"robot fell at t={d.time:.2f}s"); time.sleep(2); reset()

            sleep = m.opt.timestep - (time.perf_counter() - t_wall)
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    main()
