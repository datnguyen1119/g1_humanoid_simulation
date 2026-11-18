"""Open the cart-pole in the MuJoCo viewer with no control (ctrl = 0).

Usage:
    .venv/bin/python scripts/view.py              # start upright, tilted 0.1 rad -> falls and swings
    .venv/bin/python scripts/view.py --key down   # hanging at rest
    .venv/bin/python scripts/view.py --key up     # exactly upright (unstable equilibrium)
"""
import argparse
import pathlib
import time

import mujoco
import mujoco.viewer

MODEL = pathlib.Path(__file__).resolve().parents[1] / "model" / "cartpole.xml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="up_tilt", choices=["down", "up", "up_tilt"])
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(args.key).id)
    print(f"nq={model.nq} nv={model.nv} nu={model.nu}, start keyframe '{args.key}'")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_print = 0.0
        while viewer.is_running():
            t0 = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            if data.time - last_print >= 1.0:
                last_print = data.time
                x, th = data.qpos
                print(f"t={data.time:5.1f}s  x={x:+.3f} m  theta={th:+.3f} rad")
            dt = model.opt.timestep - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
