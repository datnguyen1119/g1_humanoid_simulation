"""Plot a log written by run_g1_idqp_logged.py.

Usage (from humanoid_g1_control/):
    .venv/bin/python sim/plot_g1_log.py                 # out/idqp_log.npz
    .venv/bin/python sim/plot_g1_log.py --tag _push3    # out/idqp_log_push3.npz
    .venv/bin/python sim/plot_g1_log.py --show          # also open a window

Shading on every panel: red = QP infeasible (torque held), orange = inaccurate / max iter, grey = push.
"""
import argparse
import csv
import pathlib

import matplotlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def spans(t, mask):
    """Contiguous [t_start, t_end] intervals where mask is true."""
    if not mask.any():
        return []
    dt = np.median(np.diff(t)) if len(t) > 1 else 0.002
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    return [(t[a], t[b - 1] + dt) for a, b in zip(idx[::2], idx[1::2])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = ROOT / "out" / f"idqp_log{args.tag}.npz"
    L = np.load(path, allow_pickle=True)
    t = L["t"]
    sv = L["status_val"]
    infeas = L["used_fallback"].astype(bool)
    warn = (~infeas) & (sv != 1)
    push = np.any(L["push"] != 0, axis=1)
    names = list(L["act_names"])

    fig, ax = plt.subplots(7, 1, figsize=(11, 15), sharex=True)
    # 1. CoM vs reference
    for k, (lab, col) in enumerate([("x", "C0"), ("y", "C1"), ("z", "C2")]):
        ax[0].plot(t, L["com"][:, k] - L["com_ref"][0, k], col, label=f"c_{lab}")
        ax[0].plot(t, L["com_ref"][:, k] - L["com_ref"][0, k], col, ls="--", lw=0.8)
    ax[0].set_ylabel("CoM - c0 (m)\n(dashed: ref)"); ax[0].legend(ncol=3, fontsize=8)
    # 2. CoM error and CoP margin
    ax[1].plot(t, L["com_err"] * 1e3, label="CoM error")
    ax[1].plot(t, L["cop_margin"] * 1e3, label="CoP margin to polygon edge")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_ylabel("mm"); ax[1].legend(fontsize=8)
    # 3. momentum
    ax[2].plot(t, L["lin_mom"][:, 0], label="p_x"); ax[2].plot(t, L["lin_mom"][:, 1], label="p_y")
    ax[2].set_ylabel("p (kg m/s)"); ax[2].legend(fontsize=8, loc="upper left")
    a2 = ax[2].twinx()
    a2.plot(t, L["ang_mom"][:, 0], "C3", lw=0.8, label="L_x"); a2.plot(t, L["ang_mom"][:, 1], "C4", lw=0.8, label="L_y")
    a2.set_ylabel("L about CoM (N m s)"); a2.legend(fontsize=8, loc="upper right")
    # 4. torque usage: max and the 3 joints that get closest to their limit
    ax[3].plot(t, L["tau_ratio_max"], "k", lw=1, label="max over joints")
    worst = np.argsort(-L["tau_ratio"].max(0))[:3]
    for j in worst:
        ax[3].plot(t, L["tau_ratio"][:, j], lw=0.8, label=names[j])
    ax[3].axhline(1, color="r", lw=0.5); ax[3].set_ylim(0, 1.1)
    ax[3].set_ylabel("|tau| / tau_max"); ax[3].legend(fontsize=8, ncol=4)
    # 5. friction (QP) and slip (sim)
    if "fric_foot_max" in L:
        ax[4].plot(t, L["fric_foot_max"], label="QP: per foot |sum f_t|/(mu sum f_z)")
    ax[4].plot(t, L["fric_ratio_max"], lw=0.5, alpha=0.5, label="QP: per corner")
    ax[4].plot(t, L["slip_sim"], lw=0.8, label="sim: max |f_t|/(mu f_n)")
    ax[4].axhline(1, color="r", lw=0.5); ax[4].set_ylim(0, 1.1)
    ax[4].set_ylabel("friction use"); ax[4].legend(fontsize=8)
    # 6. vertical force per foot
    ax[5].plot(t, L["fz_sim"][:, 0], "C0", label="L sim"); ax[5].plot(t, L["fz_qp"][:, 0], "C0--", lw=0.8, label="L QP")
    ax[5].plot(t, L["fz_sim"][:, 1], "C1", label="R sim"); ax[5].plot(t, L["fz_qp"][:, 1], "C1--", lw=0.8, label="R QP")
    ax[5].set_ylabel("f_z (N)"); ax[5].legend(fontsize=8, ncol=4)
    # 7. QP solver
    ax[6].semilogy(t, L["iter"], label="iterations")
    ax[6].semilogy(t, L["solve_ms"], label="solve ms")
    ax[6].set_ylabel("QP"); ax[6].set_xlabel("t (s)"); ax[6].legend(fontsize=8)

    for a in ax:
        for s, e in spans(t, push):
            a.axvspan(s, e, color="grey", alpha=0.15, lw=0)
        for s, e in spans(t, warn):
            a.axvspan(s, e, color="orange", alpha=0.3, lw=0)
        for s, e in spans(t, infeas):
            a.axvspan(s, e, color="red", alpha=0.3, lw=0)
    title = (f"{path.name}: mu={float(L['mu'])}, tau_scale={float(L['tau_scale'])}, push_scale={float(L['push_scale'])} | "
             f"infeasible {int(infeas.sum())} steps, inaccurate {int(warn.sum())}")
    ax[0].set_title(title + "\nred: QP infeasible (tau held), orange: inaccurate/max iter, grey: push", fontsize=9)
    fig.tight_layout()
    out = ROOT / "out" / f"idqp_plot{args.tag}.png"
    fig.savefig(out, dpi=110)
    print("plot:", out)

    ev = ROOT / "out" / f"idqp_events{args.tag}.csv"
    if ev.exists():
        with open(ev) as fh:
            rows = list(csv.DictReader(fh))
        kinds = {}
        for r in rows:
            kinds[r["event"]] = kinds.get(r["event"], 0) + 1
        print("events:", ", ".join(f"{k} x{v}" for k, v in sorted(kinds.items())))
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
