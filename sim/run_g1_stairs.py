"""G1 up and down stairs (step-to gait, piecewise LIPM + ID-QP) with log, events and PASS/FAIL report.

Usage (from humanoid_g1_control/):
    .venv/bin/python sim/run_g1_stairs.py                              # viewer, 5 cm steps
    .venv/bin/python sim/run_g1_stairs.py --headless --video           # no window, out/stairs.mp4
    .venv/bin/python sim/run_g1_stairs.py --rise 0.08 --headless --tag _r08
    .venv/bin/python sim/run_g1_stairs.py --rise 0.10 --clearance 0.05 --headless --tag _r10
    .venv/bin/python sim/plot_g1_stairs.py --tag _r08                  # side view + time plots
    .venv/bin/python sim/stairs_terrain.py --rise 0.08 --view          # look at the scene only

Outputs in out/: stairs_log<tag>.npz, stairs_events<tag>.csv, stairs<tag>.mp4 (headless --video).
Press Delete (or Backspace / the Reset button) in the viewer to restart; each run is saved as ..._runN.
"""
import argparse
import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from run_g1_idqp_logged import Session, run_headless, run_viewer  # noqa: E402
from stairs_core import StairsConfig, StairsController  # noqa: E402

PASS_COM_ERR = 0.015      # m


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--video", action="store_true", help="(headless) write out/stairs<tag>.mp4")
    ap.add_argument("--rise", type=float, default=0.05, help="step height (m)")
    ap.add_argument("--tread", type=float, default=0.30, help="step depth (m)")
    ap.add_argument("--n-up", type=int, default=3)
    ap.add_argument("--n-down", type=int, default=3)
    ap.add_argument("--landing", type=float, default=0.60, help="top platform length after the last step up (m)")
    ap.add_argument("--t-ss", type=float, default=0.6, help="single support time (s)")
    ap.add_argument("--t-ds", type=float, default=0.3, help="double support time (s)")
    ap.add_argument("--clearance", type=float, default=0.04, help="swing sole above the higher tread (m)")
    ap.add_argument("--crouch", type=float, default=0.03, help="CoM lowered during the stairs (m)")
    ap.add_argument("--T", type=float, default=None, help="headless: simulated time (default: whole route)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


def report(session, ctrl):
    """PASS/FAIL table from the log of the run that just ended."""
    lg = session.logger
    r = lg.rows
    if not r["t"]:
        return
    events = [e[1] for e in lg.events]
    n_plan = len(ctrl.steps)
    fell = "FALL" in events
    margins = [min(m_toe, m_heel) for _, _, m_toe, m_heel in ctrl.touchdowns]
    com_err = float(np.max(r["com_err"]))
    checks = [
        ("all steps done", lg.n_steps_done == n_plan and not fell, f"{lg.n_steps_done} / {n_plan} touchdowns"
         + (", FELL" if fell else "")),
        ("no infeasible QP", lg.n_infeasible == 0, f"{lg.n_infeasible} infeasible QPs"),
        ("no scuffing", "SWING_SCUFF" not in events, f"{events.count('SWING_SCUFF')} SWING_SCUFF"),
        ("feet on treads", bool(margins) and min(margins) >= ctrl.cfg.min_edge_margin,
         f"min sole margin to an edge {min(margins) * 1e3:.0f} mm" if margins else "no touchdown"),
        ("CoM tracking", com_err < PASS_COM_ERR, f"max CoM error {com_err * 1e3:.1f} mm"),
        ("torque within limits", float(np.max(r["tau_ratio_max"])) < 1.0,
         f"max |tau|/tau_max {np.max(r['tau_ratio_max']):.2f}"),
    ]
    print("\n--- stairs report (rise %.3f m, tread %.2f m) ---" % (ctrl.cfg.rise, ctrl.cfg.tread))
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<22} {detail}")
    # where the effort goes: worst joints by peak torque usage
    tr = np.array(r["tau_ratio"])
    worst = np.argsort(-tr.max(0))[:3]
    print("peak torque use:", ", ".join(f"{ctrl.act_names[j]} {tr[:, j].max():.2f}" for j in worst))
    zerr = np.array(r["zmp_err"], dtype=float)
    if np.isfinite(zerr).any():
        print("max ZMP error (CoP - ZMP ref): %.0f mm" % (np.nanmax(zerr) * 1e3))


def main():
    args = parse()
    cfg = StairsConfig(rise=args.rise, tread=args.tread, n_up=args.n_up, n_down=args.n_down,
                       landing=args.landing, t_ss=args.t_ss, t_ds=args.t_ds,
                       clearance=args.clearance, crouch=args.crouch)
    try:
        ctrl = StairsController(cfg)
    except ValueError as e:                       # plan check failed: report before simulating
        sys.exit(str(e))
    args.T = args.T or ctrl.T_TOTAL
    args.push_scale = 1.0
    print(f"stairs: {cfg.n_up} up x {cfg.rise:.3f} m, landing, {cfg.n_down} down | tread {cfg.tread} m | "
          f"{len(ctrl.stations)} stations, {len(ctrl.steps)} steps (step-to) | T_ss={cfg.t_ss}s T_ds={cfg.t_ds}s | "
          f"route lasts {ctrl.T_TOTAL:.1f} s")
    print(f"plan check: sole margin to edges >= {ctrl.plan_check['min_edge_margin'] * 1e3:.0f} mm, "
          f"swing clearance >= {ctrl.plan_check['min_swing_clear'] * 1e3:.0f} mm | scene {pathlib.Path(cfg.xml).name}")
    session = Session(ctrl, args, prefix="stairs")

    def side_view(viewer):
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = ctrl.PELVIS
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.6, 75, -12

    orig_restart = session.restart

    def restart(c):                                # report each run before the viewer reset starts a new one
        report(session, ctrl)
        orig_restart(c)                            # saves the finished run (with its touchdowns), then resets
        ctrl.touchdowns.clear()
    session.restart = restart

    try:
        if args.headless:
            run_headless(args, ctrl, session.logger)
        else:
            run_viewer(args, ctrl, session, cam_setup=side_view)
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        if session.logger.status:
            report(session, ctrl)
            session.save()


if __name__ == "__main__":
    main()
