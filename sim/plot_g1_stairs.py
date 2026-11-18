"""Plot a stairs log written by run_g1_stairs.py: side view (x-z) and time histories.

Usage (from humanoid_g1_control/):
    .venv/bin/python sim/plot_g1_stairs.py              # out/stairs_log.npz
    .venv/bin/python sim/plot_g1_stairs.py --tag _r08
    .venv/bin/python sim/plot_g1_stairs.py --show

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

    path = ROOT / "out" / f"stairs_log{args.tag}.npz"
    L = np.load(path, allow_pickle=True)
    t = L["t"]
    infeas = L["used_fallback"].astype(bool)
    warn = (~infeas) & (L["status_val"] != 1)
    ss = L["phase_id"] == 2
    names = list(L["act_names"])
    qpos = L["qpos"]
    tp = L["tp"]

    fig = plt.figure(figsize=(12, 16))
    gs = fig.add_gridspec(7, 1, height_ratios=[1.7, 1, 1, 1, 1, 1, 1])

    # 1. side view: terrain, planned footsteps, CoM, feet (ankle origins from the log via swing_pos is
    #    only for the swing foot, so use the corner positions? not logged -> use qpos-independent data)
    a = fig.add_subplot(gs[0])
    a.fill_between(L["terrain_x"], 0, L["terrain_h"], color="tan", alpha=0.6, step=None, label="stairs")
    so = L["sole_off"]
    for k, (sw, x, y, z) in enumerate(L["footsteps"]):
        a.plot([x - 0.055, x + 0.125], [z - 0.035] * 2, color="C0" if sw == 0 else "C1", lw=3, alpha=0.6)
    a.plot(L["com_plan"][:, 0, 0], L["z_plan"], "b--", lw=1, label="CoM plan")
    a.plot(L["com"][:, 0], L["com"][:, 2], "b", lw=1.3, label="CoM")
    sp = L["swing_pos"]; sr = L["swing_ref"]
    for k, c in ((0, "C0"), (1, "C1")):
        mk = L["swing_foot"] == k
        a.plot(np.where(mk, sr[:, 0], np.nan), np.where(mk, sr[:, 2] - 0.035, np.nan), c + "--", lw=0.8)
        a.plot(np.where(mk, sp[:, 0], np.nan), np.where(mk, sp[:, 2] - 0.035, np.nan), c, lw=1.2,
               label=f"{'LR'[k]} sole (swing), dashed = ref")
    a.set_aspect("equal"); a.set_xlabel("x (m)"); a.set_ylabel("z (m)")
    a.set_ylim(-0.05, max(0.9, float(np.nanmax(L["com"][:, 2])) + 0.05))
    a.legend(fontsize=7, ncol=4, loc="upper left")
    a.set_title(f"{path.name}: rise {float(L['rise']):.3f} m, tread {float(L['tread']):.2f} m | "
                f"infeasible {int(infeas.sum())} QPs | CoM x {L['com'][0, 0]:.2f} -> {L['com'][-1, 0]:.2f} m",
                fontsize=9)

    ax = [fig.add_subplot(gs[i]) for i in range(1, 7)]
    for b in ax[1:]:
        b.sharex(ax[0])
    # 2. CoM z
    ax[0].plot(tp, L["z_plan"], "b--", lw=1, label="CoM z plan")
    ax[0].plot(t, L["com"][:, 2], "b", lw=1, label="CoM z")
    ax[0].plot(t, L["com_ref"][:, 2], "c:", lw=1, label="CoM z ref (QP)")
    ax[0].set_ylabel("z (m)"); ax[0].legend(fontsize=7, ncol=3)
    # 3. CoM / ZMP y
    ax[1].plot(tp, L["zmp_plan"][:, 1], "m--", lw=1, label="ZMP plan")
    ax[1].plot(t, L["cop_sim"][:, 1], "y", lw=0.7, label="CoP measured")
    ax[1].plot(t, L["com"][:, 1], "b", lw=1, label="CoM y")
    ax[1].set_ylabel("y (m)"); ax[1].legend(fontsize=7, ncol=3)
    # 4. errors
    ax[2].plot(t, L["com_err"] * 1e3, label="CoM error")
    ax[2].plot(t, L["swing_err"] * 1e3, label="swing error")
    ax[2].plot(t, np.clip(L["zmp_err"] * 1e3, 0, 200), lw=0.6, label="|CoP - ZMP ref| (clip 200)")
    ax[2].plot(t, np.clip(L["cop_margin"] * 1e3, -20, 100), lw=0.6, label="CoP margin (clip)")
    ax[2].axhline(0, color="k", lw=0.5); ax[2].set_ylabel("mm"); ax[2].legend(fontsize=7, ncol=4)
    # 5. torque usage of the worst joints
    tr = L["tau_ratio"]
    for j in np.argsort(-tr.max(0))[:5]:
        ax[3].plot(t, tr[:, j], lw=0.8, label=names[j])
    ax[3].axhline(1, color="r", lw=0.5); ax[3].set_ylim(0, 1.15); ax[3].set_ylabel("|tau|/tau_max")
    ax[3].legend(fontsize=7, ncol=5)
    # 6. vertical forces
    ax[4].plot(t, L["fz_sim"][:, 0], "C0", lw=0.8, label="fz L sim"); ax[4].plot(t, L["fz_qp"][:, 0], "C0--", lw=0.6)
    ax[4].plot(t, L["fz_sim"][:, 1], "C1", lw=0.8, label="fz R sim"); ax[4].plot(t, L["fz_qp"][:, 1], "C1--", lw=0.6)
    ax[4].set_ylabel("N"); ax[4].legend(fontsize=7, ncol=2)
    # 7. QP
    ax[5].semilogy(t, L["iter"], label="iterations"); ax[5].semilogy(t, L["solve_ms"], label="solve ms")
    ax[5].set_xlabel("t (s)"); ax[5].legend(fontsize=7)
    for b in ax:
        for s0, e0 in spans(t, ss):
            b.axvspan(s0, e0, color="tab:blue", alpha=0.06, lw=0)
        for s0, e0 in spans(t, warn):
            b.axvspan(s0, e0, color="orange", alpha=0.3, lw=0)
        for s0, e0 in spans(t, infeas):
            b.axvspan(s0, e0, color="red", alpha=0.3, lw=0)
    fig.tight_layout()
    out = ROOT / "out" / f"stairs_plot{args.tag}.png"
    fig.savefig(out, dpi=105)
    print("plot:", out)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
