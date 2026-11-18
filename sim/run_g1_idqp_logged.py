"""ID-QP balance for the G1 with a full state log, events and on-screen status.

Usage (from humanoid_g1_control/):
    .venv/bin/python sim/run_g1_idqp_logged.py                      # live viewer, default scenario
    .venv/bin/python sim/run_g1_idqp_logged.py --headless --video   # no window, write an mp4
    # scenarios that make the QP infeasible (see the events printed in the terminal):
    .venv/bin/python sim/run_g1_idqp_logged.py --push-scale 3                 # 180 N push: infeasible at ~7.0 s, falls
    .venv/bin/python sim/run_g1_idqp_logged.py --tau-scale 0.25               # weak motors: infeasible after the push
    .venv/bin/python sim/run_g1_idqp_logged.py --mu 0.15 --push-scale 2       # low friction + 120 N push
    (--mu 0.15 alone stays feasible: the 60 N pushes need little friction)

Outputs in out/ (suffix from --tag):
    idqp_log.npz      every control step (state, contacts, QP info)  -> sim/plot_g1_log.py
    idqp_events.csv   events only: QP infeasible (with cause), torque/friction limits, CoP near edge, ...
    idqp.mp4          with --video (headless)

Viewer: text at the top left, 3 scrolling plots at the bottom, markers on the floor:
    green/orange/red polygon = support polygon (orange: CoP near the edge, red: QP infeasible in the last 0.3 s)
    blue sphere = CoM on the floor, light blue = CoM reference, yellow square = CoP (sim),
    orange lines = contact forces planned by the QP, red arrow = push.
Double-click the torso and Ctrl + right-drag to push it yourself.
Press Delete (or Backspace / the Reset button) to restart the scenario; each run is saved as ..._runN.
"""
import argparse
import pathlib
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from idqp_core import ROOT, Config, IDQPController  # noqa: E402
from idqp_monitor import Logger, LiveFigures, add_markers, overlay_text, robot_state  # noqa: E402

DISPLAY_HZ = 20.0       # text / plots update rate
RENDER_HZ = 60.0        # viewer sync rate
FALL_Z = 0.4            # pelvis height below which the robot has fallen
# Optional controller hooks (used by the stairs controller; absent -> the defaults below):
#   ctrl.is_fallen() -> bool            default: pelvis z < FALL_Z
#   ctrl.camera_lookat(st) -> xyz       video camera target; default: CoM xy at z = 0.55
#   ctrl.video_camera -> (dist, azim, elev)


def robot_fell(ctrl):
    return ctrl.is_fallen() if hasattr(ctrl, "is_fallen") else ctrl.d.qpos[2] < FALL_Z
KEY_DELETE = 261        # GLFW key code
PANEL_W = 480           # video: width of the status-text column (px)


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headless", action="store_true", help="no viewer window")
    ap.add_argument("--video", action="store_true", help="(headless) write out/idqp<tag>.mp4")
    ap.add_argument("--T", type=float, default=10.0, help="headless: simulated time (s)")
    ap.add_argument("--mu", type=float, default=0.5, help="friction coefficient in the QP (default 0.5)")
    ap.add_argument("--floor-mu", type=float, default=None, help="also set the real floor friction in MuJoCo")
    ap.add_argument("--tau-scale", type=float, default=1.0, help="scale actuator torque limits")
    ap.add_argument("--push-scale", type=float, default=1.0, help="scale the scripted pushes (60 N)")
    ap.add_argument("--tag", default="", help="suffix for output file names")
    ap.add_argument("--quiet", action="store_true", help="do not print events to the terminal")
    return ap.parse_args()


def make_controller(args):
    cfg = Config(mu=args.mu, tau_scale=args.tau_scale, push_scale=args.push_scale)
    ctrl = IDQPController(cfg)
    if args.floor_mu is not None:
        ctrl.m.geom_friction[:, 0] = args.floor_mu      # MuJoCo uses the max of the two geoms' friction
    return ctrl


def control_step(ctrl, logger):
    """One 2 ms step: solve, apply torque and push, log. Returns the state dict."""
    d = ctrl.d
    tau, info = ctrl.solve(d.time)
    if info["status"] == "interrupted":          # Ctrl+C arrived while OSQP was solving
        raise KeyboardInterrupt
    if info["kind"] == "error":
        raise RuntimeError(f"QP error at t={d.time:.3f}: {info['status']}")
    d.ctrl[:] = tau
    F = ctrl.push_force(d.time)
    d.xfrc_applied[ctrl.TORSO, :3] = F                  # zero when no scripted push
    st = robot_state(ctrl, info, tau, F)
    logger.add(st)
    mujoco.mj_step(ctrl.m, d)
    return st


def run_headless(args, ctrl, logger):
    d, m = ctrl.d, ctrl.m
    frames, renderer = [], None
    if args.video:
        from PIL import Image, ImageDraw, ImageFont
        m.vis.global_.offwidth, m.vis.global_.offheight = max(960, m.vis.global_.offwidth), max(544, m.vis.global_.offheight)
        renderer = mujoco.Renderer(m, 544, 960)   # multiples of 16 for the video codec
        cam = mujoco.MjvCamera(); cam.distance, cam.azimuth, cam.elevation = getattr(ctrl, "video_camera", (2.4, 140, -15))
        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", 14)
        except OSError:
            font = ImageFont.load_default()
        every = int(round(1 / (30 * m.opt.timestep)))
    step, last_print = 0, -1.0
    while d.time < args.T:
        st = control_step(ctrl, logger)
        if renderer is not None and step % every == 0:
            cam.lookat[:] = ctrl.camera_lookat(st) if hasattr(ctrl, "camera_lookat") else [st["com"][0], st["com"][1], 0.55]
            renderer.update_scene(d, camera=cam)
            add_markers(renderer.scene, ctrl, st, d.time - logger.last_infeasible_t < 0.3)
            scene_img = renderer.render()
            # status text in its own column on the left, so it never hides the robot
            panel = Image.new("RGB", (PANEL_W, scene_img.shape[0]), (0, 0, 0))
            draw = ImageDraw.Draw(panel)
            labels, values = overlay_text(st, logger)
            red = st["kind"] != "ok"
            for i, (a, b) in enumerate(zip(labels.split("\n"), values.split("\n"))):
                b_col = (255, 80, 80) if (red and a == "QP") else (255, 255, 255)
                draw.text((8, 8 + 18 * i), a, fill=(170, 170, 170), font=font)
                draw.text((140, 8 + 18 * i), b, fill=b_col, font=font)
            frames.append(np.hstack([np.asarray(panel), scene_img]))
        step += 1
        if d.time - last_print >= 1.0:
            last_print = d.time
            print(f"t={d.time:5.2f}s  com_err={st['com_err'] * 1e3:5.1f} mm  {st['status']}", flush=True)
        if robot_fell(ctrl):
            logger.event(d.time, "FALL", f"pelvis z={d.qpos[2]:.2f} m")
            break
    if frames:
        import imageio
        path = logger.out / f"{logger.prefix}{args.tag}.mp4"
        imageio.mimsave(path, frames, fps=30, quality=8)
        print("video:", path)


def run_viewer(args, ctrl, session, cam_setup=None):
    """Live viewer. Reset with Delete (or the viewer's Reset button / Backspace): the run so far is saved
    as its own log (..._run1, _run2, ...) and the scenario starts again from t = 0."""
    import mujoco.viewer
    d, m = ctrl.d, ctrl.m
    reset_request = {"flag": False}

    def on_key(keycode):                    # runs in the viewer thread: only set a flag
        if keycode == KEY_DELETE:          # every letter is already a viewer shortcut (R = reflection)
            reset_request["flag"] = True

    def new_run():
        nonlocal figs, last_disp, last_sync, fallen, st, last_time
        figs = LiveFigures()
        last_disp, last_sync, fallen, st, last_time = -1.0, 0.0, False, None, 0.0

    figs = last_disp = last_sync = fallen = st = last_time = None
    new_run()
    print("viewer: press Delete (or Backspace / the Reset button in the left panel) to restart the scenario")
    with mujoco.viewer.launch_passive(m, d, key_callback=on_key) as viewer:
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.6, 140, -15
        if cam_setup is not None:
            cam_setup(viewer)
        while viewer.is_running():
            # the viewer's own reset sets d.time back to 0 (and the pose to qpos0): treat it as a reset too
            if reset_request["flag"] or d.time < last_time - 1e-9:
                reset_request["flag"] = False
                with viewer.lock():
                    session.restart(ctrl)           # save this run, new logger, robot back to q_nom
                    viewer.user_scn.ngeom = 0
                    viewer.clear_texts(); viewer.clear_figures()
                new_run()
                viewer.sync()
                continue
            logger = session.logger
            if not fallen:
                st = control_step(ctrl, logger)
                if d.time - last_disp >= 1 / DISPLAY_HZ:
                    last_disp = d.time
                    figs.push(st)
                    labels, values = overlay_text(st, logger)
                    labels += f"\nrun"; values += f"\n{session.run}   (Delete = reset)"
                    viewer.set_texts((mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, labels, values))
                    figs.draw(viewer)
                if robot_fell(ctrl):
                    logger.event(d.time, "FALL", f"pelvis z={d.qpos[2]:.2f} m (paused, press Delete to restart)")
                    fallen = True
            last_time = d.time
            if time.perf_counter() - last_sync >= 1 / RENDER_HZ or fallen:
                if st is not None:
                    with viewer.lock():
                        viewer.user_scn.ngeom = 0
                        add_markers(viewer.user_scn, ctrl, st, d.time - logger.last_infeasible_t < 0.3)
                viewer.sync()
                last_sync = time.perf_counter()
                if fallen:
                    time.sleep(1 / RENDER_HZ)


class Session:
    """Holds the current logger; numbers runs when the viewer is reset."""

    def __init__(self, ctrl, args, prefix="idqp"):
        self.args, self.run, self.prefix = args, 1, prefix
        self.logger = Logger(ctrl, ROOT / "out", verbose=not args.quiet, prefix=prefix)

    def tag(self):
        # a single run keeps the plain name; after a reset every run gets _runN
        return self.args.tag + (f"_run{self.run}" if self.run > 1 or getattr(self, "multi", False) else "")

    def save(self):
        npz, ev = self.logger.save(self.tag())
        print(f"summary (run {self.run}):", self.logger.summary())
        print("log:", npz)
        print("events:", ev)

    def restart(self, ctrl):
        self.multi = True
        if self.logger.status:
            self.save()
        print(f"\n===== reset -> run {self.run + 1} =====")
        self.run += 1
        ctrl.reset()
        self.logger = Logger(ctrl, ROOT / "out", verbose=not self.args.quiet, prefix=self.prefix)


def main():
    args = parse()
    ctrl = make_controller(args)
    session = Session(ctrl, args)
    print(f"mu={args.mu}  tau_scale={args.tau_scale}  push_scale={args.push_scale}"
          + (f"  floor_mu={args.floor_mu}" if args.floor_mu is not None else ""))
    try:
        if args.headless:
            run_headless(args, ctrl, session.logger)
        else:
            run_viewer(args, ctrl, session)
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        if session.logger.status:
            session.save()


if __name__ == "__main__":
    main()
