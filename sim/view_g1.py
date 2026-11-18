"""Load Unitree G1 into MuJoCo and hold a standing pose with joint PD control.

Usage:
    .venv/bin/python sim/view_g1.py               # 29-DoF, pelvis hung by an elastic band
    .venv/bin/python sim/view_g1.py --no-band     # free-standing (will fall without a balancer)
    .venv/bin/python sim/view_g1.py --dof 23
    .venv/bin/python sim/view_g1.py --passive     # no control at all, robot goes limp
"""
import argparse
import pathlib
import time

import mujoco
import mujoco.viewer
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
G1_DIR = ROOT / "third_party" / "unitree_mujoco" / "unitree_robots" / "g1"

# Slightly bent-knee standing pose (rad), keyed by actuator name. Others default to 0.
STAND_POSE = {
    "left_hip_pitch": -0.1, "left_knee": 0.3, "left_ankle_pitch": -0.2,
    "right_hip_pitch": -0.1, "right_knee": 0.3, "right_ankle_pitch": -0.2,
    "left_shoulder_roll": 0.2, "left_elbow": 0.9,
    "right_shoulder_roll": -0.2, "right_elbow": 0.9,
}


def gains(name: str) -> tuple[float, float]:
    if any(k in name for k in ("hip", "knee")):
        return 150.0, 4.0
    if "ankle" in name:
        return 40.0, 2.0
    if "waist" in name:
        return 200.0, 5.0
    return 40.0, 1.0  # arms / wrists


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dof", choices=["23", "29"], default="29")
    ap.add_argument("--no-band", action="store_true", help="disable pelvis elastic band")
    ap.add_argument("--passive", action="store_true", help="no joint control")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(G1_DIR / f"scene_{args.dof}dof.xml"))
    data = mujoco.MjData(model)

    # Map each actuator to its joint's qpos/qvel address.
    act_names = [model.actuator(i).name for i in range(model.nu)]
    jnt_ids = model.actuator_trnid[:, 0]
    qadr = model.jnt_qposadr[jnt_ids]
    vadr = model.jnt_dofadr[jnt_ids]
    q_des = np.array([STAND_POSE.get(n, 0.0) for n in act_names])
    kp, kd = map(np.array, zip(*(gains(n) for n in act_names)))
    lo, hi = model.actuator_ctrlrange.T

    # Initial state: pelvis at standing height, joints at the target pose.
    data.qpos[2] = 0.78
    data.qpos[qadr] = q_des
    mujoco.mj_forward(model, data)

    pelvis = model.body("pelvis").id
    band_anchor = np.array([0.0, 0.0, 1.0])
    band_on = not args.no_band

    def key_cb(keycode: int) -> None:
        nonlocal band_on
        if keycode == ord("B"):
            band_on = not band_on
            print(f"elastic band {'ON' if band_on else 'OFF'}")

    print(f"Loaded G1 {args.dof}-DoF: nq={model.nq} nu={model.nu}. Press B to toggle band.")
    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
        while viewer.is_running():
            t0 = time.time()
            if not args.passive:
                tau = kp * (q_des - data.qpos[qadr]) - kd * data.qvel[vadr]
                data.ctrl[:] = np.clip(tau, lo, hi)
            data.xfrc_applied[pelvis] = 0.0
            if band_on:  # spring-damper pulling pelvis toward the anchor
                dx = band_anchor - data.xpos[pelvis]
                data.xfrc_applied[pelvis, :3] = 500.0 * dx - 50.0 * data.qvel[0:3]  # pelvis is the free-joint root
            mujoco.mj_step(model, data)
            viewer.sync()
            dt = model.opt.timestep - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
