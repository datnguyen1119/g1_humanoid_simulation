"""G1 walking with DCM feedback and step adjustment (dcm_core.py), flat floor, with log / events / report.

Usage (from humanoid_g1_control/):
    .venv/bin/python sim/run_g1_dcm.py                                  # viewer, 8 steps of 10 cm
    .venv/bin/python sim/run_g1_dcm.py --push --push-f 40               # same push that makes run_g1_walk.py fall
    .venv/bin/python sim/run_g1_dcm.py --push --no-step-adjust          # DCM feedback only (CoP), no stepping
    .venv/bin/python sim/run_g1_dcm.py --headless --video --push --tag _push40
    .venv/bin/python sim/plot_g1_dcm.py --tag _push40

Outputs in out/: dcm_log<tag>.npz, dcm_events<tag>.csv, dcm<tag>.mp4 (headless --video).

Viewer markers (on the floor):
    red sphere      = CP / DCM measured, xi = c + c'/w
    pink sphere     = CP reference xi_ref
    yellow square   = CoP measured in the simulation
    red line        = CoP -> CP: the DCM runs away from the CoP along this line (xi' = w (xi - CoP))
    orange disc     = CoP commanded by the DCM controller (red when clipped by the support polygon)
    magenta sphere  = ZMP reference of the plan
    green box       = landing point of the swing foot (moves when the step is adjusted)
    cyan / grey boxes = footsteps (current plan / nominal plan)
Press Delete (or Backspace / the Reset button) to restart; each run is saved as ..._runN.

The shared runner (run_g1_idqp_logged.py) is reused unchanged: this script swaps four of its names
(robot_state, Logger, add_markers, LiveFigures) for versions that add the DCM fields, events and markers.
"""
import argparse
import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_g1_idqp_logged as base  # noqa: E402
from dcm_core import DCMConfig, DCMWalkController  # noqa: E402
from idqp_monitor import LiveFigures, Logger, add_markers, robot_state  # noqa: E402

DCM_KEYS = ["dcm", "dcm_ref", "dcm_err", "V", "r_ref", "r_cmd_raw", "r_cmd", "cop_sat", "step_shift", "dcm_mode", "p_cmd"]


# ---------------------------------------------------------------------- DCM additions to the shared monitor
def dcm_robot_state(ctrl, info, tau, push):
    st = robot_state(ctrl, info, tau, push)
    st.update({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ctrl.dcm_state.items()})
    return st


class DCMLogger(Logger):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rows.update({k: [] for k in DCM_KEYS})
        self.n_sat = 0

    def add(self, st):
        super().add(st)
        for k in DCM_KEYS:
            self.rows[k].append(st[k])
        t, warn = st["t"], self.ctrl.cfg.dcm_err_warn
        e = st["dcm_err"]
        self._edge("dcm", e > warn, e < 0.7 * warn, t, "DCM_ERR_HIGH", f"|xi - xi_ref| = {e * 1e3:.0f} mm", "DCM_ERR_OK")
        sat = bool(st["cop_sat"])
        self.n_sat += sat
        r, rr = st["r_cmd"], st["r_cmd_raw"]
        self._edge("copsat", self._debounced("copsat", sat), not sat, t, "COP_CMD_SAT",
                   f"wanted CoP {np.linalg.norm(rr - r) * 1e3:.0f} mm outside the support polygon (inset)", "COP_CMD_OK")


def dcm_add_markers(scn, ctrl, st, recent_infeasible=False, **kw):
    add_markers(scn, ctrl, st, recent_infeasible, **kw)
    ctrl.extra_markers(scn, st)


class DCMFigures(LiveFigures):
    SPECS = [("DCM err (mm)", lambda s: s["dcm_err"] * 1e3, None),
             ("CoP cmd clipped (1 = yes)", lambda s: s["cop_sat"], (0, 1.1)),
             ("tau / tau_max", lambda s: s["tau_ratio_max"], (0, 1.1))]


base.robot_state, base.Logger, base.add_markers, base.LiveFigures = dcm_robot_state, DCMLogger, dcm_add_markers, DCMFigures


# ---------------------------------------------------------------------- CLI and report
def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--video", action="store_true", help="(headless) write out/dcm<tag>.mp4")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--step-len", type=float, default=0.10)
    ap.add_argument("--t-ss", type=float, default=0.5)
    ap.add_argument("--t-ds", type=float, default=0.2)
    ap.add_argument("--k-dcm", type=float, default=6.0, help="DCM error decay rate (1/s)")
    ap.add_argument("--cop-inset", type=float, default=0.0, help="CoP command margin inside the support polygon (m)")
    ap.add_argument("--no-step-adjust", action="store_true", help="DCM feedback through the CoP only")
    ap.add_argument("--w-step", type=float, default=1.0, help="cost of moving a step (vs 1.0 for moving the CoP)")
    ap.add_argument("--push", action="store_true", help="lateral push on the torso (default: after the 3rd step)")
    ap.add_argument("--push-f", type=float, default=40.0, help="push force (N, +y = to the left)")
    ap.add_argument("--push-t", type=float, default=None, help="push start time (s)")
    ap.add_argument("--push-dur", type=float, default=0.15)
    ap.add_argument("--T", type=float, default=None, help="headless: simulated time (default: whole walk)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


def report(session, ctrl):
    lg = session.logger
    r = lg.rows
    if not r["t"]:
        return None
    events = [e[1] for e in lg.events]
    fell = "FALL" in events
    n = len(r["t"])
    e = np.array(r["dcm_err"])
    shifts = np.array([[dx, dy] for _, _, dx, dy in ctrl.commits]) if ctrl.commits else np.zeros((0, 2))
    checks = [
        ("all steps done", lg.n_steps_done == len(ctrl.steps) and not fell,
         f"{lg.n_steps_done} / {len(ctrl.steps)} touchdowns" + (f", FELL at {r['t'][-1]:.2f} s" if fell else "")),
        ("no infeasible QP", lg.n_infeasible == 0, f"{lg.n_infeasible} infeasible QPs"),
        ("DCM tracking", e.max() < ctrl.cfg.dcm_err_warn, f"max |xi - xi_ref| {e.max() * 1e3:.1f} mm"),
        ("torque within limits", float(np.max(r["tau_ratio_max"])) < 1.0, f"max |tau|/tau_max {np.max(r['tau_ratio_max']):.2f}"),
    ]
    print(f"\n--- DCM walk report (k={ctrl.cfg.k_dcm}, step adjust {'on' if ctrl.cfg.step_adjust else 'off'}"
          + (f", push {ctrl.cfg.pushes[0][2][1]:+.0f} N at {ctrl.cfg.pushes[0][0]:.2f} s" if ctrl.cfg.pushes else "") + ") ---")
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<22} {detail}")
    print(f"CoP command clipped {lg.n_sat / n * 100:.1f} % of the steps | max |CoP - r_cmd| "
          f"{np.nanmax(np.linalg.norm(np.array(r['cop_sim']) - np.array(r['r_cmd']), axis=1)) * 1e3:.0f} mm")
    if len(shifts):
        j = np.argmax(np.linalg.norm(shifts, axis=1))
        print(f"largest landing shift: step {ctrl.commits[j][1] + 1} ({shifts[j, 0] * 1e3:+.0f}, {shifts[j, 1] * 1e3:+.0f}) mm")
    return not fell


def main():
    args = parse()
    cfg = DCMConfig(n_steps=args.steps, step_len=args.step_len, t_ss=args.t_ss, t_ds=args.t_ds,
                    k_dcm=args.k_dcm, cop_inset=args.cop_inset, step_adjust=not args.no_step_adjust,
                    w_step=args.w_step, push=args.push, push_f=args.push_f, push_t=args.push_t, push_dur=args.push_dur)
    ctrl = DCMWalkController(cfg)
    ctrl.video_camera = (2.6, 115, -35)          # look down a bit more: the CoP / CP markers are on the floor
    args.T = args.T or ctrl.T_TOTAL
    args.push_scale = 1.0
    print(f"DCM walk: {cfg.n_steps} steps x {cfg.step_len} m, T_ss={cfg.t_ss}s T_ds={cfg.t_ds}s | w={ctrl.omega:.2f} 1/s, "
          f"k={cfg.k_dcm} -> CoP gain 1+k/w = {1 + cfg.k_dcm / ctrl.omega:.2f} | step adjust {'on' if cfg.step_adjust else 'off'}"
          + (f" | push {cfg.push_f:+.0f} N x {cfg.push_dur}s at t={cfg.pushes[0][0]:.2f}s" if cfg.pushes else ""))
    session = base.Session(ctrl, args, prefix="dcm")

    def follow_pelvis(viewer):
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = ctrl.PELVIS
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.4, 120, -25

    orig_restart = session.restart

    def restart(c):                            # report each run before the viewer reset starts a new one
        report(session, ctrl)
        orig_restart(c)
    session.restart = restart

    try:
        if args.headless:
            base.run_headless(args, ctrl, session.logger)
        else:
            base.run_viewer(args, ctrl, session, cam_setup=follow_pelvis)
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        if session.logger.status:
            report(session, ctrl)
            session.save()


if __name__ == "__main__":
    main()
