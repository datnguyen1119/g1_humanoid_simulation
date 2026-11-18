"""Step 7: swing-up then LQR.

Pass: from the bottom, reaches the top and holds 10 s; no chattering between modes;
cart within the rail, force within ctrlrange; save a video with mediapy.

Uses rho from outputs/step5_lqr.npz if step 5 has been run, else a default.
Options: --no-video to skip rendering.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # import modules from cartpole_control/

import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from controllers import SWINGUP_GAINS, HybridController, design_lqr, upright_error  # noqa: E402
from params import P  # noqa: E402
from sim_utils import OUT_DIR, load, report, run, save_fig  # noqa: E402

DEFAULT_RHO = 29.0  # value measured by step 5 with LQR_Q, LQR_R


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    model, data = load()
    K, S = design_lqr()
    npz = OUT_DIR / "step5_lqr.npz"
    rho = float(np.load(npz)["rho"]) if npz.exists() else DEFAULT_RHO
    print(f"rho = {rho:.2f} ({'from step 5' if npz.exists() else 'default'})")
    ctrl = HybridController(K, S, rho, SWINGUP_GAINS)

    modes = []

    def controller(x):
        u = ctrl(x)
        modes.append(ctrl.mode)
        return u

    # Optional video: render at 30 fps while simulating
    frames, renderer, cam = [], None, None
    if not args.no_video:
        try:
            renderer = mujoco.Renderer(model, height=480, width=854)
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [0, 0, 0]; cam.distance = 4.0; cam.azimuth = 90; cam.elevation = -10
        except Exception as e:  # no OpenGL context available
            print(f"video disabled: {e}")
    every = int(round(1 / 30 / model.opt.timestep))

    def on_step():
        if renderer is not None and int(round(data.time / model.opt.timestep)) % every == 0:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())

    T = 20.0
    log = run(model, data, controller, np.array([0.0, 0.05, 0.0, 0.0]), T, on_step=on_step)
    t, X, U = log["t"], log["x"], log["u"]
    is_lqr = np.array([m == "lqr" for m in modes])

    n_switch = int(np.sum(is_lqr[1:] != is_lqr[:-1]))
    t_switch = float(t[np.argmax(is_lqr)]) if is_lqr.any() else np.inf
    e_th = np.abs([upright_error(x)[1] for x in X])
    # "reached" = last time the pole was outside |theta - pi| < 0.05; it must stay inside afterwards
    outside = e_th >= 0.05
    t_reach = float(t[outside][-1]) if outside.any() else 0.0
    held = is_lqr.any() and (T - t_reach) >= 10.0

    report("reaches top and holds 10 s", held,
           f"switch to LQR at t = {t_switch:.2f} s, inside |theta - pi| < 0.05 from t = {t_reach:.2f} s "
           f"({T - t_reach:.1f} s held)")
    report("no chattering", n_switch == 1, f"{n_switch} mode switch(es)")
    report("cart within rail", np.abs(X[:, 0]).max() < P.x_max, f"max |x| = {np.abs(X[:, 0]).max():.2f} m")
    report("force within ctrlrange", np.abs(U).max() <= P.u_max + 1e-9, f"max |u| = {np.abs(U).max():.1f} N")

    fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    ax[0].plot(t, X[:, 1]); ax[0].axhline(np.pi, color="k", ls="--", lw=0.5); ax[0].set_ylabel("theta (rad)")
    ax[1].plot(t, X[:, 0]); ax[1].set_ylabel("x (m)")
    ax[2].plot(t, U); ax[2].set_ylabel("u (N)"); ax[2].set_xlabel("t (s)")
    for a in ax:
        a.axvspan(t_switch, T, color="green", alpha=0.08)
    ax[0].set_title("swing-up then LQR (green: LQR mode)")
    print("plot:", save_fig(fig, "step7_full.png"))

    if frames:
        import mediapy
        path = OUT_DIR / "step7_full.mp4"
        mediapy.write_video(str(path), frames, fps=30)
        print("video:", path)


if __name__ == "__main__":
    main()
