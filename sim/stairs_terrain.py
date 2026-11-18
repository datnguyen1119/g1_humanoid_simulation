"""Stairs terrain (up, landing, down) shared by the scene builder and the stairs planner.

The same Terrain object builds the MuJoCo scene and answers planner questions (height under a point,
distance to the nearest step edge), so the plan and the simulated geometry cannot disagree.

Build / view the scene on its own (from humanoid_g1_control/):
    .venv/bin/python sim/stairs_terrain.py                    # writes sim/scenes/g1_stairs.xml
    .venv/bin/python sim/stairs_terrain.py --rise 0.08 --view # build and open the MuJoCo viewer

Profile along x (robot starts at x = 0, facing +x):
    floor --approach--| up 1 | up 2 | up 3 |--- landing ---| down 3 | down 2 | down 1 |--- floor
"""
from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
MENAGERIE_SCENE = ROOT / "third_party/mujoco_menagerie/unitree_g1/scene.xml"
SCENE_DIR = pathlib.Path(__file__).resolve().parent / "scenes"


@dataclass
class Terrain:
    rise: float = 0.05        # height of one step (m)
    tread: float = 0.30       # depth of one step (m)
    n_up: int = 3
    n_down: int = 3
    landing: float = 0.60     # length of the top platform (m)
    approach: float = 0.35    # x of the first riser (robot starts at x = 0)
    width: float = 1.2        # stair width along y (m)

    def __post_init__(self):
        # flat segments (x_start, x_end, height); the floor extends to +-inf at both ends
        segs, x = [], self.approach
        for k in range(1, self.n_up + 1):
            segs.append((x, x + self.tread, k * self.rise)); x += self.tread
        top = self.n_up * self.rise
        segs.append((x, x + self.landing, top)); x += self.landing
        self.rise_down = top / self.n_down               # equals `rise` when n_down == n_up
        for k in range(1, self.n_down):                  # the n_down-th step down lands on the floor
            segs.append((x, x + self.tread, top - k * self.rise_down)); x += self.tread
        self.x_end = x
        self.segments = segs
        # real edges only: boundaries where the height changes (top step and landing share a height)
        cand = sorted({s[0] for s in segs} | {s[1] for s in segs})
        self.edges = [x for x in cand if abs(self.h(x - 1e-6) - self.h(x + 1e-6)) > 1e-9]

    # ------------------------------------------------------------------ queries
    def h(self, x: float) -> float:
        """Ground height at x (floor = 0)."""
        for x0, x1, hh in self.segments:
            if x0 <= x < x1:
                return hh
        return 0.0

    def segment_at(self, x: float):
        """(x_start, x_end, height) of the flat piece under x; the floor pieces are unbounded."""
        for s in self.segments:
            if s[0] <= x < s[1]:
                return s
        if x < self.approach:
            return (-np.inf, self.approach, 0.0)
        return (self.x_end, np.inf, 0.0)

    def treads(self):
        """Flat pieces in walking order, including the floor before and after."""
        return [(-np.inf, self.approach, 0.0)] + list(self.segments) + [(self.x_end, np.inf, 0.0)]

    def profile(self, x0=-0.3, x1=None, n=2000):
        """Sampled (x, h) for plotting."""
        x1 = self.x_end + 0.6 if x1 is None else x1
        xs = np.linspace(x0, x1, n)
        return xs, np.array([self.h(x) for x in xs])

    # ------------------------------------------------------------------ MuJoCo scene
    def build_spec(self) -> mujoco.MjSpec:
        spec = mujoco.MjSpec.from_file(str(MENAGERIE_SCENE))
        spec.meshdir = str(MENAGERIE_SCENE.parent / spec.meshdir)   # absolute: the xml can live anywhere
        spec.modelname = "g1_stairs"
        spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1088   # room for wide videos
        for i, (x0, x1, hh) in enumerate(self.segments):
            g = spec.worldbody.add_geom()                 # solid block from the floor up to the tread
            g.name = f"stair_{i}"
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = [(x1 - x0) / 2, self.width / 2, hh / 2]
            g.pos = [(x0 + x1) / 2, 0.0, hh / 2]
            shade = 0.45 + 0.1 * (i % 2)
            g.rgba = [shade + 0.15, shade + 0.05, shade - 0.05, 1]
            g.friction = [1.0, 0.005, 0.0001]
        for i, x in enumerate(self.edges):                # thin visual stripe on every step edge (no contact)
            hh = max(self.h(x - 1e-6), self.h(x + 1e-6))
            g = spec.worldbody.add_geom()
            g.name = f"edge_{i}"
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = [0.006, self.width / 2, 0.0008]
            g.pos = [x, 0.0, hh + 0.0008]
            g.rgba = [0.95, 0.75, 0.1, 1]
            g.contype = 0; g.conaffinity = 0
        return spec

    def write_scene(self, path: pathlib.Path | None = None) -> pathlib.Path:
        path = path or SCENE_DIR / "g1_stairs.xml"
        path.parent.mkdir(parents=True, exist_ok=True)
        spec = self.build_spec()
        spec.compile()                                    # fail here, not later, if something is wrong
        path.write_text(spec.to_xml())
        return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rise", type=float, default=0.05)
    ap.add_argument("--tread", type=float, default=0.30)
    ap.add_argument("--n-up", type=int, default=3)
    ap.add_argument("--n-down", type=int, default=3)
    ap.add_argument("--landing", type=float, default=0.60)
    ap.add_argument("--view", action="store_true", help="open the scene in the MuJoCo viewer")
    a = ap.parse_args()
    terr = Terrain(rise=a.rise, tread=a.tread, n_up=a.n_up, n_down=a.n_down, landing=a.landing)
    path = terr.write_scene()
    print("scene:", path)
    for s in terr.segments:
        print(f"  x {s[0]:.2f} .. {s[1]:.2f}  h = {s[2]:.3f} m")
    if a.view:
        import mujoco.viewer
        m = mujoco.MjModel.from_xml_path(str(path))
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.viewer.launch(m, d)


if __name__ == "__main__":
    main()
