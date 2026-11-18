"""Push-recovery sweep on flat ground: preview walking (walk_core) vs DCM (dcm_core) without / with step adjustment vs ZMP MPC (mpc_core).

Lateral push on the torso, push_dur s, started after the 3rd step + 0.3 s (t = 3.4 s with the default timing),
i.e. in the single support on the LEFT foot:
    F > 0 (+y, to the left)  pushes the DCM outward of the stance foot: the swing (right) foot cannot catch it
    F < 0 (-y, to the right) pushes the DCM toward the swing foot: one adjusted step can catch it

Usage (from humanoid_g1_control/):
    .venv/bin/python sim/sweep_g1_push.py                              # default forces, 6 processes
    .venv/bin/python sim/sweep_g1_push.py --forces -80 -60 -40 40 60 --jobs 4
Writes out/push_sweep.csv and prints a table (time of the fall, or "ok").
"""
import argparse
import csv
import os
import pathlib
import sys
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTROLLERS = ["preview", "dcm_nostep", "dcm_step", "mpc"]


def run(job):
    import mujoco
    import numpy as np
    from dcm_core import DCMConfig, DCMWalkController
    from mpc_core import MPCConfig, MPCWalkController
    from walk_core import WalkConfig, WalkController
    name, force, dur, kd, inset = job
    if name == "preview":
        ctrl = WalkController(WalkConfig(push=True, push_f=force, push_dur=dur))
    elif name == "mpc":
        ctrl = MPCWalkController(MPCConfig(push=True, push_f=force, push_dur=dur))
    else:
        ctrl = DCMWalkController(DCMConfig(push=True, push_f=force, push_dur=dur, step_adjust=(name == "dcm_step"),
                                                   k_dcm=kd, cop_inset=inset))
    d, m = ctrl.d, ctrl.m
    n_inf, t_fall = 0, None
    while d.time < ctrl.T_TOTAL:
        tau, info = ctrl.solve(d.time)
        n_inf += info["kind"] == "infeasible"
        d.ctrl[:] = tau
        d.xfrc_applied[ctrl.TORSO, :3] = ctrl.push_force(d.time)
        mujoco.mj_step(m, d)
        if d.qpos[2] < 0.4:
            t_fall = d.time
            break
    shift = max((float(np.hypot(dx, dy)) for _, _, dx, dy in getattr(ctrl, "commits", [])), default=0.0)
    return dict(controller=name, force=force, survived=t_fall is None, t_fall=t_fall, infeasible=n_inf,
                max_step_shift_mm=round(shift * 1e3))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forces", type=float, nargs="+", default=[-80, -60, -50, -40, -30, -20, 20, 30, 40, 50, 60])
    ap.add_argument("--dur", type=float, default=0.15)
    ap.add_argument("--controllers", nargs="+", default=CONTROLLERS, choices=CONTROLLERS)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--k-dcm", type=float, default=6.0, help="DCM controllers: error decay rate (1/s)")
    ap.add_argument("--cop-inset", type=float, default=0.0,
                    help="DCM controllers: r_cmd margin inside the support polygon (m); < 0 lets the CMP leave it")
    args = ap.parse_args()
    jobs = [(c, f, args.dur, args.k_dcm, args.cop_inset) for f in args.forces for c in args.controllers]
    with Pool(args.jobs) as pool:
        res = pool.map(run, jobs)
    out = ROOT / "out" / "push_sweep.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
    print(f"\nk_dcm={args.k_dcm} cop_inset={args.cop_inset} | lateral push {args.dur} s at t = 3.4 s (single support on the LEFT foot); cell = fall time or ok [max step shift]")
    print(f"{'F (N)':>7} | " + " | ".join(f"{c:^18}" for c in args.controllers))
    for f in args.forces:
        cells = []
        for c in args.controllers:
            r = next(r for r in res if r["controller"] == c and r["force"] == f)
            s = "ok" if r["survived"] else f"fall {r['t_fall']:.2f}s"
            if c == "dcm_step":
                s += f" [{r['max_step_shift_mm']} mm]"
            cells.append(f"{s:^18}")
        print(f"{f:>+7.0f} | " + " | ".join(cells))
    print("csv:", out)


if __name__ == "__main__":
    main()
