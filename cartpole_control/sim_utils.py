"""Small helpers shared by the step scripts: load the model, run a closed loop, save plots."""
import pathlib
from typing import Callable, Optional

import matplotlib

matplotlib.use("Agg")  # save figures to files, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from params import MODEL_PATH  # noqa: E402

OUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"


def load() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    return model, mujoco.MjData(model)


def get_state(data: mujoco.MjData) -> np.ndarray:
    return np.concatenate([data.qpos, data.qvel]).copy()


def set_state(model, data, x: np.ndarray) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[:] = x[:2]
    data.qvel[:] = x[2:]
    mujoco.mj_forward(model, data)


def run(model, data, controller: Callable[[np.ndarray], float], x0: np.ndarray, T: float,
        push: Optional[Callable[[float], float]] = None,
        on_step: Optional[Callable[[], None]] = None) -> dict:
    """Closed-loop MuJoCo rollout. push(t) returns a horizontal force (N) applied to the cart."""
    set_state(model, data, x0)
    cart = model.body("cart").id
    n = int(round(T / model.opt.timestep))
    log = {"t": np.empty(n), "x": np.empty((n, 4)), "u": np.empty(n)}
    for k in range(n):
        x = get_state(data)
        u = controller(x)
        data.ctrl[0] = u
        data.xfrc_applied[cart, 0] = push(data.time) if push else 0.0
        log["t"][k], log["x"][k], log["u"][k] = data.time, x, u
        mujoco.mj_step(model, data)
        if on_step:
            on_step()
    return log


def save_fig(fig, name: str) -> pathlib.Path:
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def report(name: str, ok: bool, detail: str) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok
