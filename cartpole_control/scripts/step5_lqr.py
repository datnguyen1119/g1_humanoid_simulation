"""Step 5: LQR balance at the top.

Pass: from theta = pi +- 0.2 rad, settles within 3 s without hitting +-20 N.
Also plot the measured region of attraction on the (theta, thetad) plane.
Writes outputs/step5_lqr.npz (K, S, rho) for step 7.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # import modules from cartpole_control/

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from controllers import (LQR_Q, LQR_R, design_lqr, lqr_continuous, lqr_control,  # noqa: E402
                         lqr_discrete, upright_error)
from linearize import X_UP, linearize_analytic, linearize_mujoco_step  # noqa: E402
from params import P  # noqa: E402
from sim_utils import OUT_DIR, load, report, run, save_fig  # noqa: E402


def settle_time(log, th_tol=0.01, x_tol=0.02) -> float:
    e = np.array([upright_error(x) for x in log["x"]])
    bad = (np.abs(e[:, 1]) > th_tol) | (np.abs(e[:, 0]) > x_tol)
    return float(log["t"][bad][-1]) if bad.any() else 0.0


def main() -> None:
    model, data = load()
    A, B = linearize_analytic()
    K, S = design_lqr()
    print("K (continuous) =", np.round(K, 3))

    # Continuous vs discrete LQR: for small dt the gains should be almost the same
    Ad, Bd = linearize_mujoco_step(model, data, X_UP, 0.0)
    dt = model.opt.timestep
    Kd, _ = lqr_discrete(Ad, Bd, LQR_Q * dt, LQR_R * dt)   # scale cost by dt to match the integral
    print("K (discrete)   =", np.round(Kd, 3))

    # 1) Main test: theta = pi +- 0.2
    fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    ok = True
    for th0 in (0.2, -0.2):
        log = run(model, data, lambda x: lqr_control(x, K), np.array([0, np.pi + th0, 0, 0]), 5.0)
        ts, umax = settle_time(log), np.abs(log["u"]).max()
        ok &= report(f"theta0 = pi{th0:+.1f}", ts < 3.0 and umax < P.u_max,
                     f"settle {ts:.2f} s, max |u| = {umax:.1f} N")
        ax[0].plot(log["t"], [upright_error(x)[1] for x in log["x"]], label=f"theta0 = pi{th0:+.1f}")
        ax[1].plot(log["t"], log["x"][:, 0])
        ax[2].plot(log["t"], log["u"])
    ax[0].set_ylabel("theta - pi (rad)"); ax[0].legend()
    ax[1].set_ylabel("x (m)"); ax[2].set_ylabel("u (N)"); ax[2].set_xlabel("t (s)")
    print("plot:", save_fig(fig, "step5_lqr_response.png"))

    # 2) Q/R trade-off: faster convergence costs more force
    print("\nQ/R sweep (theta0 = pi + 0.2):")
    for q_th, r in [(10, 1.0), (50, 0.1), (200, 0.01)]:
        Kq, _ = lqr_continuous(A, B, np.diag([10, q_th, 1, 1]), np.array([[r]]))
        log = run(model, data, lambda x: lqr_control(x, Kq), np.array([0, np.pi + 0.2, 0, 0]), 6.0)
        print(f"  Q_theta = {q_th:4d}, R = {r:5.2f}: settle {settle_time(log):.2f} s, "
              f"max |u| = {np.abs(log['u']).max():5.1f} N")

    # 3) Push test: 30 N on the cart for 0.1 s at t = 1 s
    log = run(model, data, lambda x: lqr_control(x, K), X_UP.copy(), 6.0,
              push=lambda t: 30.0 if 1.0 <= t < 1.1 else 0.0)
    e_end = np.abs(upright_error(log["x"][-1])[1])
    report("recovers from 30 N x 0.1 s push", e_end < 0.01,
           f"max |theta err| {np.abs([upright_error(x)[1] for x in log['x']]).max():.3f} rad")

    # 4) Region of attraction: grid of (theta_err, thetad), cart starts at rest at x = 0
    ths = np.linspace(-1.2, 1.2, 25)
    thds = np.linspace(-6, 6, 25)
    success = np.zeros((len(thds), len(ths)), dtype=bool)
    cost = np.zeros_like(success, dtype=float)
    for i, thd in enumerate(thds):
        for j, th in enumerate(ths):
            x0 = np.array([0.0, np.pi + th, 0.0, thd])
            log = run(model, data, lambda x: lqr_control(x, K), x0, 4.0)
            e = np.array([upright_error(x) for x in log["x"]])
            success[i, j] = np.abs(e[-1, 1]) < 0.05 and np.abs(e[:, 0]).max() < P.x_max - 0.05
            cost[i, j] = upright_error(x0) @ S @ upright_error(x0)
    # rho: largest level set of x^T S x that contains only successful grid points
    rho = float(cost[~success].min()) if (~success).any() else float(cost.max())
    np.savez(OUT_DIR / "step5_lqr.npz", K=K, S=S, rho=rho)
    print(f"\nregion of attraction: {success.mean() * 100:.0f}% of grid succeeds, rho = {rho:.2f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.contourf(ths, thds, success, levels=[-0.5, 0.5, 1.5], colors=["#e57373", "#81c784"])
    ax.contour(ths, thds, cost, levels=[rho], colors="k", linestyles="--")
    ax.set_xlabel("theta - pi (rad)"); ax.set_ylabel("thetad (rad/s)")
    ax.set_title("LQR region of attraction (green = success), dashed: x^T S x = rho")
    print("plot:", save_fig(fig, "step5_roa.png"))
    report("step 5 overall", ok, "theta0 = pi +- 0.2 settles < 3 s within force limit")


if __name__ == "__main__":
    main()
