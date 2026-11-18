"""G1 walking (ZMP preview + ID-QP) with full state log, events and on-screen status.

Usage (from humanoid_g1_control/):
    .venv/bin/python sim/run_g1_walk.py                          # live viewer, 8 steps of 10 cm
    .venv/bin/python sim/run_g1_walk.py --headless --video       # no window, write out/walk.mp4
    .venv/bin/python sim/run_g1_walk.py --steps 12 --step-len 0.15 --t-ss 0.45
    .venv/bin/python sim/run_g1_walk.py --push --push-f 40       # lateral push after the 3rd step
    .venv/bin/python sim/plot_g1_walk.py                         # plot the last log

Outputs in out/ (suffix from --tag): walk_log.npz, walk_events.csv, walk.mp4 (headless --video).

Viewer markers: cyan boxes = planned footsteps, magenta = ZMP reference, yellow square = measured CoP,
blue = CoM on the floor, cyan sphere = swing-foot target, polygon = support polygon (red: QP infeasible).
Press Delete (or Backspace / the Reset button) to restart the walk; each run is saved as ..._runN.
"""
import argparse
import pathlib
import sys

import mujoco

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from run_g1_idqp_logged import Session, run_headless, run_viewer  # noqa: E402
from walk_core import WalkConfig, WalkController  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headless", action="store_true", help="no viewer window")
    ap.add_argument("--video", action="store_true", help="(headless) write out/walk<tag>.mp4")
    ap.add_argument("--steps", type=int, default=8, help="number of steps (the last one closes the feet)")
    ap.add_argument("--step-len", type=float, default=0.10, help="x distance between the feet after a step (m)")
    ap.add_argument("--t-ss", type=float, default=0.5, help="single support time (s)")
    ap.add_argument("--t-ds", type=float, default=0.2, help="double support time (s)")
    ap.add_argument("--step-h", type=float, default=0.05, help="swing apex height (m)")
    ap.add_argument("--push", action="store_true", help="lateral push on the torso after the 3rd step")
    ap.add_argument("--push-f", type=float, default=40.0, help="push force (N, +y = to the left)")
    ap.add_argument("--mu", type=float, default=0.5, help="friction coefficient in the QP")
    ap.add_argument("--tau-scale", type=float, default=1.0, help="scale actuator torque limits")
    ap.add_argument("--T", type=float, default=None, help="headless: simulated time (default: whole walk)")
    ap.add_argument("--tag", default="", help="suffix for output file names")
    ap.add_argument("--quiet", action="store_true", help="do not print events to the terminal")
    return ap.parse_args()


def main():
    args = parse()
    cfg = WalkConfig(n_steps=args.steps, step_len=args.step_len, t_ss=args.t_ss, t_ds=args.t_ds,
                     step_h=args.step_h, push=args.push, push_f=args.push_f, mu=args.mu, tau_scale=args.tau_scale)
    ctrl = WalkController(cfg)
    args.T = args.T or ctrl.T_TOTAL
    args.push_scale = 1.0
    print(f"{cfg.n_steps} steps x {cfg.step_len} m, T_ss={cfg.t_ss}s T_ds={cfg.t_ds}s, h={cfg.step_h} m | "
          f"walk lasts {ctrl.T_TOTAL:.2f} s | CoM height zc={ctrl.ZC:.3f} m"
          + (f" | push {cfg.push_f} N at t={cfg.pushes[0][0]:.2f}s" if cfg.pushes else ""))
    session = Session(ctrl, args, prefix="walk")

    def follow_pelvis(viewer):                 # camera tracks the robot; mouse still rotates / zooms
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = ctrl.PELVIS
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.4, 120, -15

    try:
        if args.headless:
            run_headless(args, ctrl, session.logger)
        else:
            run_viewer(args, ctrl, session, cam_setup=follow_pelvis)
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        if session.logger.status:
            session.save()


if __name__ == "__main__":
    main()
