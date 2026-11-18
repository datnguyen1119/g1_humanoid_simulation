"""Step 6: energy-shaping swing-up with collocated PFL.

Pass: from the bottom with theta = 0.05, energy error E - E_top -> 0 and the cart stays within +-2 m.
E is the pole energy 1/2 J thd^2 - m_p g l cos(th) (see dynamics.pole_energy); E_top = m_p g l.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # import modules from cartpole_control/

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from controllers import SWINGUP_GAINS, swingup_control  # noqa: E402
from dynamics import pole_energy  # noqa: E402
from params import P  # noqa: E402
from sim_utils import load, report, run, save_fig  # noqa: E402


def main() -> None:
    model, data = load()
    E_top = P.m_p * P.g * P.l
    log = run(model, data, lambda x: swingup_control(x, **SWINGUP_GAINS),
              np.array([0.0, 0.05, 0.0, 0.0]), 15.0)
    t, X = log["t"], log["x"]
    E_err = np.array([pole_energy(x) for x in X]) - E_top

    tail = np.abs(E_err[t > t[-1] - 2.0]).max()          # last 2 s
    report("E - E_top -> 0", tail < 0.01 * E_top, f"max |E - E_top| in last 2 s = {tail:.1e} J "
           f"({tail / E_top * 100:.2f}% of E_top)")
    xmax = np.abs(X[:, 0]).max()
    report("cart within +-2 m", xmax < P.x_max, f"max |x| = {xmax:.2f} m")
    print(f"max |u| = {np.abs(log['u']).max():.1f} N")

    fig, ax = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
    ax[0].plot(t, E_err); ax[0].axhline(0, color="k", lw=0.5); ax[0].set_ylabel("E - E_top (J)")
    ax[1].plot(t, X[:, 1]); ax[1].axhline(np.pi, color="k", ls="--", lw=0.5); ax[1].set_ylabel("theta (rad)")
    ax[2].plot(t, X[:, 0]); ax[2].set_ylabel("x (m)")
    ax[3].plot(t, log["u"]); ax[3].set_ylabel("u (N)"); ax[3].set_xlabel("t (s)")
    print("plot:", save_fig(fig, "step6_swingup.png"))


if __name__ == "__main__":
    main()
