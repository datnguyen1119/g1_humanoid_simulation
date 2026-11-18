"""Plot a walking log written by run_g1_walk.py.

Usage (from humanoid_g1_control/):
    .venv/bin/python sim/plot_g1_walk.py                # out/walk_log.npz
    .venv/bin/python sim/plot_g1_walk.py --tag _push    # out/walk_log_push.npz
    .venv/bin/python sim/plot_g1_walk.py --show

Shading: light blue = single support, red = QP infeasible (torque held), orange = inaccurate, grey = push.
"""
import argparse
import pathlib

import matplotlib
import numpy as np

from plot_g1_log import spans

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = ROOT / "out" / f"walk_log{args.tag}.npz"
    L = np.load(path, allow_pickle=True)
    t = L["t"]
    infeas = L["used_fallback"].astype(bool)
    warn = (~infeas) & (L["status_val"] != 1)
    push = np.any(L["push"] != 0, axis=1)
    ss = L["phase_id"] == 2
    tp, com_plan, zmp_plan = L["tp"], L["com_plan"], L["zmp_plan"]

    fig = plt.figure(figsize=(12, 15))
    gs = fig.add_gridspec(7, 1, height_ratios=[1, 1, 1, 1, 1, 1, 1.6])
    ax = [fig.add_subplot(gs[i]) for i in range(6)]
    for a in ax[1:]:
        a.sharex(ax[0])

    # 1-2. ZMP and CoM along x and y: plan vs measured
    for k, lab in enumerate("xy"):
        a = ax[k]
        a.plot(tp, zmp_plan[:, k], "m--", lw=1, label="ZMP plan (preview)")
        a.plot(t, L["zmp_ref"][:, k], "m:", lw=1, label="ZMP ref")
        a.plot(t, L["cop_sim"][:, k], "y", lw=0.8, alpha=0.9, label="CoP measured (sim)")
        a.plot(tp, com_plan[:, 0, k], "b--", lw=1, label="CoM plan")
        a.plot(t, L["com"][:, k], "b", lw=1.2, label="CoM")
        a.set_ylabel(f"{lab} (m)")
        a.legend(fontsize=7, ncol=5, loc="upper left")
    # 3. errors
    ax[2].plot(t, L["com_err"] * 1e3, label="CoM error")
    ax[2].plot(t, L["swing_err"] * 1e3, label="swing foot error")
    ax[2].plot(t, np.clip(L["zmp_err"] * 1e3, 0, 200), lw=0.7, label="|CoP - ZMP ref| (clipped 200)")
    ax[2].set_ylabel("mm"); ax[2].legend(fontsize=7, ncol=3)
    # 4. swing foot height and vertical force per foot
    ax[3].plot(t, L["swing_ref"][:, 2], "c--", lw=1, label="swing z ref")
    ax[3].plot(t, L["swing_pos"][:, 2], "c", lw=1, label="swing z")
    ax[3].plot(t, L["fz_sim"][:, 0] / 1e4, "C0", lw=0.8, label="fz L / 1e4")
    ax[3].plot(t, L["fz_sim"][:, 1] / 1e4, "C1", lw=0.8, label="fz R / 1e4")
    ax[3].set_ylabel("m"); ax[3].legend(fontsize=7, ncol=4)
    # 5. limits
    ax[4].plot(t, L["tau_ratio_max"], "k", lw=1, label="max |tau|/tau_max")
    ax[4].plot(t, L["fric_foot_max"], lw=0.8, label="friction use per foot (QP)")
    ax[4].plot(t, L["fric_ratio_max"], lw=0.4, alpha=0.4, label="per corner (QP)")
    ax[4].axhline(1, color="r", lw=0.5); ax[4].set_ylim(0, 1.1)
    ax[4].legend(fontsize=7, ncol=2)
    # 6. QP
    ax[5].semilogy(t, L["iter"], label="iterations"); ax[5].semilogy(t, L["solve_ms"], label="solve ms")
    ax[5].set_xlabel("t (s)"); ax[5].legend(fontsize=7)

    for a in ax:
        for s, e in spans(t, ss):
            a.axvspan(s, e, color="tab:blue", alpha=0.06, lw=0)
        for s, e in spans(t, push):
            a.axvspan(s, e, color="grey", alpha=0.25, lw=0)
        for s, e in spans(t, warn):
            a.axvspan(s, e, color="orange", alpha=0.3, lw=0)
        for s, e in spans(t, infeas):
            a.axvspan(s, e, color="red", alpha=0.3, lw=0)

    # 7. top view: footsteps, ZMP, CoM
    a = fig.add_subplot(gs[6])
    so = L["sole_off"]
    for k, (sw, x, y, z) in enumerate(L["footsteps"]):
        a.add_patch(plt.Rectangle((x + so[0] - 0.07, y + so[1] - 0.03), 0.14, 0.06, fill=False,
                                  ec="C0" if sw == 0 else "C1", lw=1))
        a.text(x + so[0], y + so[1], str(k + 1), ha="center", va="center", fontsize=7)
    for f0 in L["foot0"]:
        a.add_patch(plt.Rectangle((f0[0] + so[0] - 0.07, f0[1] + so[1] - 0.03), 0.14, 0.06, fill=False, ec="k", ls=":"))
    a.plot(zmp_plan[:, 0], zmp_plan[:, 1], "m--", lw=1, label="ZMP plan")
    a.plot(L["cop_sim"][:, 0], L["cop_sim"][:, 1], "y", lw=0.5, label="CoP measured")
    a.plot(com_plan[:, 0, 0], com_plan[:, 0, 1], "b--", lw=1, label="CoM plan")
    a.plot(L["com"][:, 0], L["com"][:, 1], "b", lw=1.2, label="CoM")
    a.set_aspect("equal"); a.set_xlabel("x (m)"); a.set_ylabel("y (m)")
    a.legend(fontsize=7, ncol=4, loc="upper left")
    a.set_title("top view (boxes: planned soles, blue = left, orange = right, dotted = start)", fontsize=9)

    n_steps = len(L["footsteps"])
    ax[0].set_title(f"{path.name}: {n_steps} steps, CoM x {L['com'][0, 0]:.3f} -> {L['com'][-1, 0]:.3f} m | "
                    f"infeasible {int(infeas.sum())} QPs, inaccurate {int(warn.sum())}\n"
                    "blue band: single support, red: QP infeasible, orange: inaccurate, grey: push", fontsize=9)
    fig.tight_layout()
    out = ROOT / "out" / f"walk_plot{args.tag}.png"
    fig.savefig(out, dpi=110)
    print("plot:", out)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
