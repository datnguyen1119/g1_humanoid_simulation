"""G1 walking with a linear ZMP MPC (mpc_core.py), flat floor, fixed footsteps, with log / events / report.

Usage (from humanoid_g1_control/):
    .venv/bin/python sim/run_g1_mpc.py                                  # viewer, 8 steps of 10 cm
    .venv/bin/python sim/run_g1_mpc.py --headless                       # report only
    .venv/bin/python sim/run_g1_mpc.py --headless --open-loop           # MPC from its own plan (~ preview control)
    .venv/bin/python sim/run_g1_mpc.py --push --push-f 40               # lateral push after the 3rd step
    .venv/bin/python sim/run_g1_mpc.py --headless --video --push --tag _push40
    .venv/bin/python sim/mpc_core.py                                    # offline check on the ideal LIPM

Outputs in out/: mpc_log<tag>.npz, mpc_events<tag>.csv, mpc<tag>.mp4 (headless --video).

Viewer markers (on the floor):
    green spheres   = ZMP predicted by the MPC over the horizon (red: on the edge of the support polygon)
    blue line       = CoM predicted by the MPC over the horizon
    red sphere      = DCM measured, xi = c + c'/w
    purple sphere   = xi_N, predicted DCM at the end of the horizon
    pink square     = terminal box xi_ref(t + N T) +- term_eps
    yellow square   = CoP measured in the simulation; magenta = ZMP reference of the plan
Press Delete (or Backspace / the Reset button) to restart; each run is saved as ..._runN.

The shared runner (run_g1_idqp_logged.py) is reused unchanged: this script swaps four of its names
(robot_state, Logger, add_markers, LiveFigures) for versions that add the MPC fields, events and markers.
"""
import argparse
import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_g1_idqp_logged as base  # noqa: E402
from idqp_monitor import LiveFigures, Logger, add_markers, robot_state  # noqa: E402
from mpc_core import STATUS, MPCConfig, MPCWalkController  # noqa: E402

MPC_KEYS = ["mpc_status", "mpc_ms", "mpc_iter", "zmp_pred", "com_pred", "zmp_active", "term_active",
            "xi_N", "xi_ref_N", "xi", "xi_ref", "dcm_err", "zmp_mpc"]


# ---------------------------------------------------------------------- MPC additions to the shared monitor
def mpc_robot_state(ctrl, info, tau, push):
    st = robot_state(ctrl, info, tau, push)
    st.update({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ctrl.mpc_state.items()})
    return st


class MPCLogger(Logger):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rows.update({k: [] for k in MPC_KEYS})

    def add(self, st):
        super().add(st)
        for k in MPC_KEYS:
            self.rows[k].append(st[k])
        t, s = st["t"], int(st["mpc_status"])
        self._edge("mpcfb", s > 0, s == 0, t, "MPC_FALLBACK", f"MPC QP: {STATUS[s]}", "MPC_OK")
        warn = self.ctrl.cfg.dcm_err_warn
        e = st["dcm_err"]
        self._edge("dcm", e > warn, e < 0.7 * warn, t, "DCM_ERR_HIGH", f"|xi - xi_ref| = {e * 1e3:.0f} mm", "DCM_ERR_OK")


def mpc_add_markers(scn, ctrl, st, recent_infeasible=False, **kw):
    add_markers(scn, ctrl, st, recent_infeasible, **kw)
    ctrl.extra_markers(scn, st)


class MPCFigures(LiveFigures):
    SPECS = [("|xi - xi_ref| (mm)", lambda s: s["dcm_err"] * 1e3, None),
             ("MPC ZMP rows at the edge", lambda s: s["zmp_active"], None),
             ("tau / tau_max", lambda s: s["tau_ratio_max"], (0, 1.1))]


base.robot_state, base.Logger, base.add_markers, base.LiveFigures = mpc_robot_state, MPCLogger, mpc_add_markers, MPCFigures


# ---------------------------------------------------------------------- CLI and report
def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--video", action="store_true", help="(headless) write out/mpc<tag>.mp4")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--step-len", type=float, default=0.10)
    ap.add_argument("--t-ss", type=float, default=0.5)
    ap.add_argument("--t-ds", type=float, default=0.2)
    ap.add_argument("--T-mpc", type=float, default=0.05, help="MPC sampling time (s)")
    ap.add_argument("--N", type=int, default=32, help="MPC horizon (samples)")
    ap.add_argument("--alpha", type=float, default=1e-6, help="jerk weight (ZMP weight gamma = 1)")
    ap.add_argument("--margin", type=float, default=0.005, help="ZMP constraint margin inside the support polygon (m)")
    ap.add_argument("--no-terminal", action="store_true", help="drop the terminal DCM constraint")
    ap.add_argument("--term-eps", type=float, default=0.01, help="terminal box half size (m)")
    ap.add_argument("--open-loop", action="store_true", help="MPC initial state from its own plan, not measured")
    ap.add_argument("--replan-dt", type=float, default=0.02, help="MPC period (s)")
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
    fell = "FALL" in [e[1] for e in lg.events]
    e = np.array(r["dcm_err"])
    n = ctrl.n_status
    ms = np.array(ctrl.mpc_ms) if ctrl.mpc_ms else np.zeros(1)
    checks = [
        ("all steps done", lg.n_steps_done == len(ctrl.steps) and not fell,
         f"{lg.n_steps_done} / {len(ctrl.steps)} touchdowns" + (f", FELL at {r['t'][-1]:.2f} s" if fell else "")),
        ("no infeasible ID-QP", lg.n_infeasible == 0, f"{lg.n_infeasible} infeasible QPs"),
        ("MPC always feasible", n[1] + n[2] == 0, f"{n[0]} ok, {n[1]} without terminal, {n[2]} kept the old plan"),
        ("DCM tracking", e.max() < ctrl.cfg.dcm_err_warn, f"max |xi - xi_ref| {e.max() * 1e3:.1f} mm"),
        ("torque within limits", float(np.max(r["tau_ratio_max"])) < 1.0, f"max |tau|/tau_max {np.max(r['tau_ratio_max']):.2f}"),
    ]
    cfg = ctrl.cfg
    print(f"\n--- MPC walk report (T={cfg.mpc_T}s N={cfg.mpc_N} alpha={cfg.alpha:g} "
          f"{'closed' if cfg.feedback else 'open'} loop, terminal {'on' if cfg.use_terminal else 'off'}"
          + (f", push {cfg.pushes[0][2][1]:+.0f} N at {cfg.pushes[0][0]:.2f} s" if cfg.pushes else "") + ") ---")
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<22} {detail}")
    zs, zm = np.array(r["cop_sim"]), np.array(r["zmp_mpc"])
    print(f"MPC solve {ms.mean():.2f} ms mean / {ms.max():.2f} ms max | ZMP rows at the edge in "
          f"{np.mean(np.array(r['zmp_active']) > 0) * 100:.1f} % of the steps | terminal at the edge "
          f"{np.mean(r['term_active']) * 100:.1f} % | max |CoP_sim - ZMP_mpc| "
          f"{np.nanmax(np.linalg.norm(zs - zm, axis=1)) * 1e3:.0f} mm")
    return not fell


def main():
    args = parse()
    cfg = MPCConfig(n_steps=args.steps, step_len=args.step_len, t_ss=args.t_ss, t_ds=args.t_ds,
                    mpc_T=args.T_mpc, mpc_N=args.N, alpha=args.alpha, zmp_margin=args.margin,
                    use_terminal=not args.no_terminal, term_eps=args.term_eps, feedback=not args.open_loop,
                    replan_dt=args.replan_dt, push=args.push, push_f=args.push_f, push_t=args.push_t,
                    push_dur=args.push_dur)
    ctrl = MPCWalkController(cfg)
    ctrl.video_camera = (2.6, 115, -35)          # look down a bit more: the MPC markers are on the floor
    args.T = args.T or ctrl.T_TOTAL
    args.push_scale = 1.0
    print(f"MPC walk: {cfg.n_steps} steps x {cfg.step_len} m, T_ss={cfg.t_ss}s T_ds={cfg.t_ds}s | "
          f"MPC T={cfg.mpc_T}s N={cfg.mpc_N} ({cfg.mpc_T * cfg.mpc_N:.2f} s) alpha/gamma={cfg.alpha / cfg.gamma:g} "
          f"every {cfg.replan_dt}s, {'closed' if cfg.feedback else 'open'} loop, terminal "
          f"{'+-%.0f mm' % (cfg.term_eps * 1e3) if cfg.use_terminal else 'off'}"
          + (f" | push {cfg.push_f:+.0f} N x {cfg.push_dur}s at t={cfg.pushes[0][0]:.2f}s" if cfg.pushes else ""))
    session = base.Session(ctrl, args, prefix="mpc")

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
