"""Push resilience of the walking controllers: largest lateral impulse J (and dv = J / m) that is survived.

A short push (default 0.1 s) on the torso during step 4 (swing RIGHT, stance LEFT foot) at several gait phases.
For each controller / direction / phase the impulse grid is run and the reported value is the
largest J such that every J' <= J on the grid survived (falls are not always monotonic in J).

Usage (from humanoid_g1_control/):
    .venv/bin/python sim/resilience_g1.py                     # preview vs DCM+step, 4 phases, +-y, J = 1..8 N s
    .venv/bin/python sim/resilience_g1.py --impulses 1 2 3 4 5 6 7 8 9 10 --jobs 8
Writes out/resilience.csv (--csv to change). Controllers: preview, dcm_nostep, dcm_step, mpc.
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
# step 4 with the default timing: DS 3.1-3.3 s, SS 3.3-3.8 s (stance LEFT)
PHASES = {"DS": 3.15, "SS early": 3.35, "SS mid": 3.50, "SS late": 3.65}


def run(job):
    import mujoco
    import numpy as np
    from dcm_core import DCMConfig, DCMWalkController
    from mpc_core import MPCConfig, MPCWalkController
    from walk_core import WalkConfig, WalkController
    name, J, sign, t_push, dur = job
    F = sign * J / dur
    if name == "preview":
        ctrl = WalkController(WalkConfig())
    elif name == "mpc":
        ctrl = MPCWalkController(MPCConfig())
    else:
        ctrl = DCMWalkController(DCMConfig(step_adjust=(name == "dcm_step")))
    ctrl.cfg.pushes = [(t_push, dur, np.array([0.0, F, 0.0]))]
    d, m = ctrl.d, ctrl.m
    while d.time < ctrl.T_TOTAL:
        tau, _ = ctrl.solve(d.time)
        d.ctrl[:] = tau
        d.xfrc_applied[ctrl.TORSO, :3] = ctrl.push_force(d.time)
        mujoco.mj_step(m, d)
        if d.qpos[2] < 0.4:
            return job, False, ctrl.mass
    return job, True, ctrl.mass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impulses", type=float, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--dur", type=float, default=0.1)
    ap.add_argument("--controllers", nargs="+", default=["preview", "dcm_step"],
                    choices=["preview", "dcm_nostep", "dcm_step", "mpc"])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--csv", default="resilience.csv", help="output file name in out/")
    args = ap.parse_args()
    jobs = [(c, J, s, t, args.dur) for c in args.controllers for s in (+1, -1)
            for t in PHASES.values() for J in args.impulses]
    with Pool(args.jobs) as pool:
        res = pool.map(run, jobs)
    mass = res[0][2]
    ok = {job: surv for job, surv, _ in res}
    rows = []
    print(f"\nlateral push {args.dur} s during step 4 (stance LEFT); robot mass {mass:.2f} kg")
    print("cell = largest J (N s) with every smaller J on the grid survived  ->  dv = J/m (m/s)")
    print(f"{'controller':<10} {'direction':<26}" + "".join(f"{p:>16}" for p in PHASES))
    for c in args.controllers:
        for s, lab in ((+1, "+y left  (outward, hard)"), (-1, "-y right (toward swing)")):
            cells = []
            for pname, t in PHASES.items():
                safe = 0.0
                for J in sorted(args.impulses):
                    if ok[(c, J, s, t, args.dur)]:
                        safe = J
                    else:
                        break
                survived = [J for J in args.impulses if ok[(c, J, s, t, args.dur)]]
                rows.append(dict(controller=c, direction=s, phase=pname, t_push=t, J_safe=safe, dv_safe=safe / mass,
                                 survived=" ".join(f"{J:g}" for J in survived)))
                cells.append(f"{safe:>4.0f} -> {safe / mass:.2f}")
            print(f"{c:<10} {lab:<26}" + "".join(f"{x:>16}" for x in cells))
    out = ROOT / "out" / args.csv
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("csv:", out, "(column 'survived' lists every J that did not fall)")


if __name__ == "__main__":
    main()
