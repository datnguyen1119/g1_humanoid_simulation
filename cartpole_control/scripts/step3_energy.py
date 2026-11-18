"""Step 3: passive simulation with your own Euler and RK4, track energy.

Pass: RK4 energy drift < 1e-3 J over 10 s; Euler drifts clearly. Plot E(t) for both.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # import modules from cartpole_control/

import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from dynamics import energy, f  # noqa: E402
from integrators import euler_step, rk4_step, simulate  # noqa: E402
from sim_utils import get_state, load, report, save_fig, set_state  # noqa: E402


def main() -> None:
    dt, T = 0.002, 10.0
    x0 = np.array([0.0, np.pi - 0.1, 0.0, 0.0])   # same as keyframe up_tilt
    E0 = energy(x0)

    trajs = {"Euler": simulate(euler_step, f, x0, 0.0, dt, T),
             "RK4": simulate(rk4_step, f, x0, 0.0, dt, T)}

    # MuJoCo's own RK4 rollout, to check our trajectory, not just energy
    model, data = load()
    set_state(model, data, x0)
    mj = [get_state(data)]
    for _ in range(len(trajs["RK4"]) - 1):
        mujoco.mj_step(model, data)
        mj.append(get_state(data))
    mj = np.array(mj)

    t = np.arange(len(trajs["RK4"])) * dt
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    drift = {}
    for name, tr in trajs.items():
        E = np.array([energy(x) for x in tr])
        drift[name] = np.abs(E - E0).max()
        ax[0].plot(t, E - E0, label=f"{name} (max |dE| = {drift[name]:.1e} J)")
        ax[1].plot(t, tr[:, 1], label=name)
    ax[1].plot(t, mj[:, 1], "k--", lw=1, label="MuJoCo")
    ax[0].set_ylabel("E(t) - E0 (J)"); ax[0].set_yscale("symlog", linthresh=1e-8); ax[0].legend()
    ax[1].set_ylabel("theta (rad)"); ax[1].set_xlabel("t (s)"); ax[1].legend()
    path = save_fig(fig, "step3_energy.png")

    report("RK4 energy drift < 1e-3 J", drift["RK4"] < 1e-3, f"{drift['RK4']:.2e} J")
    report("Euler drifts clearly", drift["Euler"] > 100 * drift["RK4"], f"{drift['Euler']:.2e} J")
    err = np.abs(trajs["RK4"] - mj).max()
    report("our RK4 trajectory matches MuJoCo RK4", err < 1e-6, f"max state err {err:.2e}")
    print(f"plot: {path}")


if __name__ == "__main__":
    main()
