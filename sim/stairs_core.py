"""G1 on stairs (up, landing, down) with a step-to gait, built on WalkController (walk_core.py).

What changes compared to flat walking:
  1. Footsteps ("step-to"): one station per tread (sole centered on it), several on the landing.
     At every station the lead foot steps first, then the trail foot joins it; the lead alternates.
  2. CoM height plan (piecewise LIPM): every phase has a target height
         z = zc - crouch + min(stance foot height, swing target height)
     reached during the double support before it. Going up the CoM rises only once the lead foot
     stands on the higher tread; going down it is lowered BEFORE the lead foot reaches down
     (the leg is too short otherwise: hip-ankle 0.60 m standing, ~0.64 m fully stretched).
     The x/y plan still comes from ZMP preview with a constant height zc - crouch (approximation:
     the vertical acceleration of the CoM is ignored in the ZMP model; its effect shows up as ZMP error).
  3. Swing foot: lift to max(z_from, z_to) + clearance, move horizontally, lower onto the target.
     Every piece is a quintic, so position, velocity and acceleration are continuous.
  4. The plan is checked before simulating: soles stay on the treads with a margin, the swing sole
     clears every step edge.
The ID-QP itself is unchanged: the CoM task simply receives a non-constant z reference.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from idqp_core import smooth
from stairs_terrain import SCENE_DIR, Terrain
from walk_core import WalkConfig, WalkController

# sole of the G1 foot relative to the ankle_roll_link origin (contact spheres +- their 5 mm radius)
SOLE_HEEL, SOLE_TOE = -0.055, 0.125      # x extent (m)
ANKLE_ABOVE_SOLE = 0.035                 # ankle frame origin height above the sole (m)


@dataclass
class StairsConfig(WalkConfig):
    # terrain
    rise: float = 0.05
    tread: float = 0.30
    n_up: int = 3
    n_down: int = 3
    landing: float = 0.60
    approach: float = 0.35
    # gait
    t_ss: float = 0.6                # longer single support fails: with a 6 cm wide foot the ankle can
    t_ds: float = 0.3                # only move the CoP +-3 cm, lateral CoM error grows ~exp(3.84 t_ss)
    t_end_ds: float = 0.6
    t_end_stand: float = 1.5
    station_gap: float = 0.15        # sole center to the step edge next to it (m)
    max_step: float = 0.30           # longest step on flat pieces (m)
    crouch: float = 0.03             # CoM lowered during the stairs, gives the legs some reach margin (m)
    clearance: float = 0.04          # swing sole above the higher of start / target tread (m)
    clearance_down: float = 0.02     # stepping down only has to clear the edge of its own tread (m)
    min_edge_margin: float = 0.02    # required sole distance to a step edge at touchdown (m)
    min_swing_clear: float = 0.015   # required swing-sole clearance above the terrain (m)
    # swing timing inside single support (fractions of t_ss)
    s_lift: float = 0.30             # sole reaches the apex height
    s_move: tuple = (0.10, 0.90)     # horizontal motion (wide window: lower foot accelerations;
                                     # check_plan verifies the sole still clears every edge)
    s_lower: float = 0.70            # sole starts going down


class StairsController(WalkController):
    def __init__(self, cfg: StairsConfig | None = None):
        cfg = cfg or StairsConfig()
        self.terrain = Terrain(rise=cfg.rise, tread=cfg.tread, n_up=cfg.n_up, n_down=cfg.n_down,
                               landing=cfg.landing, approach=cfg.approach)
        name = f"g1_stairs_r{cfg.rise:.3f}_t{cfg.tread:.2f}_{cfg.n_up}u{cfg.n_down}d.xml"
        cfg.xml = str(self.terrain.write_scene(SCENE_DIR / name))
        self.touchdowns = []            # (step index, foot, toe margin, heel margin) filled while running
        super().__init__(cfg)           # loads the stairs scene, then calls plan()

    # ================================================================== planning
    def plan(self):
        cfg, d = self.cfg, self.d
        mujoco.mj_forward(self.m, d)
        self.foot0 = [d.xpos[b].copy() for b in self.FEET]
        self.zc0 = self.c0[2]                       # standing CoM height above the floor
        self.ZC = self.zc0 - cfg.crouch             # LIPM height used by the preview controller
        self.stations = self._plan_stations()
        self.steps = self._plan_footsteps()
        self._plan_phases()                         # same phase / ZMP logic as flat walking
        self._plan_height()
        self._plan_preview()
        self._plan_pushes()
        self.plan_check = self.check_plan()

    def _pieces(self):
        """Flat pieces of the terrain in walking order, equal heights merged (top step + landing)."""
        out = []
        for x0, x1, h in self.terrain.treads():
            if out and abs(out[-1][2] - h) < 1e-9:
                out[-1] = (out[-1][0], x1, h)
            else:
                out.append((x0, x1, h))
        return out

    def piece_at(self, x):
        """Merged flat piece (x0, x1, h) under x: its ends are real step edges (or +-inf)."""
        for x0, x1, h in self._pieces():
            if x0 <= x < x1:
                return x0, x1, h
        return self._pieces()[-1]

    def _plan_stations(self):
        """Sole-center x and ground height of every station after the start pose."""
        cfg, gap = self.cfg, self.cfg.station_gap
        pieces = self._pieces()
        sole0 = self.foot0[0][0] + self.SOLE_OFF[0]
        stations = []
        for i, (x0, x1, h) in enumerate(pieces):
            if i == 0:                                   # floor before the stairs: stop gap before the riser
                xs = [x1 - gap]
                start = sole0
            elif i == len(pieces) - 1:                   # floor after: one station gap after the last edge
                xs = [x0 + gap]
                start = None
            else:                                        # tread or landing
                a, b = x0 + gap, x1 - gap
                if b - a < 1e-6:
                    xs = [(x0 + x1) / 2]
                else:
                    n = int(np.ceil((b - a) / cfg.max_step))
                    xs = list(np.linspace(a, b, n + 1))
                start = None
            if start is not None and xs[0] - start > cfg.max_step:   # long approach: extra flat steps
                n = int(np.ceil((xs[0] - start) / cfg.max_step))
                xs = list(np.linspace(start, xs[0], n + 1)[1:-1]) + xs
            stations += [(x, h) for x in xs]
        return stations

    def _plan_footsteps(self):
        """Step-to: lead foot to the station, then the trail foot next to it; the lead alternates."""
        pos = [self.foot0[0].copy(), self.foot0[1].copy()]
        steps, lead = [], 0
        for xs, h in self.stations:
            for sw in (lead, 1 - lead):
                tgt = pos[sw].copy()
                tgt[0] = xs - self.SOLE_OFF[0]
                tgt[2] = h + ANKLE_ABOVE_SOLE
                steps.append((sw, pos[sw].copy(), tgt.copy()))
                pos[sw] = tgt
            lead = 1 - lead
        return steps

    def _plan_height(self):
        """Target CoM height for every phase; z(t) moves smoothly from one target to the next."""
        cfg = self.cfg
        targets = []
        k = 0                                            # index of the step a ds / ss phase belongs to
        n_steps = len(self.steps)
        for t0, t1, kind, info in self.phases:
            if kind == "stand":
                targets.append(self.zc0)
            elif kind == "stand_end":
                targets.append(self.zc0 + self._ground_after())
            elif kind == "ds" and k >= n_steps:          # final double support: stand up straight
                targets.append(self.zc0 + self._ground_after())
            else:
                sw, p_from, p_to = self.steps[k][:3]
                stance_h = self._foot_ground(1 - sw, k)
                swing_h = p_to[2] - ANKLE_ABOVE_SOLE
                targets.append(self.zc0 - cfg.crouch + min(stance_h, swing_h))
                if kind == "ss":
                    k += 1
        self.z_targets = targets

    def _foot_ground(self, foot, k):
        """Ground height under `foot` before step k is taken."""
        z = self.foot0[foot][2]
        for sw, _, p_to in self.steps[:k]:
            if sw == foot:
                z = p_to[2]
        return z - ANKLE_ABOVE_SOLE

    def _ground_after(self):
        return max(self._foot_ground(0, len(self.steps)), self._foot_ground(1, len(self.steps)))

    # ================================================================== references
    def com_z(self, t):
        """CoM height: a step's double support and single support form one segment, over which z moves
        from the previous target to this step's target (slow vertical motion keeps z'' << g, as LIPM
        assumes). Going down, the target is reached exactly when the lead foot lands; going up, the rise
        continues while the robot stands on the higher lead foot."""
        ph = self.phases
        for i, (t0, t1, kind, info) in enumerate(ph):
            if t0 <= t < t1:
                if kind == "ds" and i + 1 < len(ph) and ph[i + 1][2] == "ss":
                    a, b, j = t0, ph[i + 1][1], i                  # ds + following ss
                elif kind == "ss" and ph[i - 1][2] == "ds":
                    a, b, j = ph[i - 1][0], t1, i - 1              # preceding ds + ss
                else:
                    a, b, j = t0, t1, i
                z_prev = self.z_targets[j - 1] if j > 0 else self.z_targets[0]
                return z_prev + (self.z_targets[i] - z_prev) * smooth((t - a) / (b - a))
        return self.z_targets[-1]

    def com_ref(self, t, h=1e-3):
        p, v, a = super().com_ref(t)
        z, zp, zm = self.com_z(t), self.com_z(t + h), self.com_z(max(t - h, 0.0))
        p[2] = z
        v[2] = (zp - zm) / (2 * h)
        a[2] = (zp - 2 * z + zm) / (h * h)
        return p, v, a

    def swing_pos(self, p_from, p_to, s):
        c = self.cfg
        s = float(np.clip(s, 0, 1))
        a, b = c.s_move
        p = p_from + (p_to - p_from) * smooth((s - a) / (b - a))
        peak = p_from[2] + c.clearance_down if p_to[2] < p_from[2] - 1e-6 else max(p_from[2], p_to[2]) + c.clearance
        if s < c.s_lift:
            z = p_from[2] + (peak - p_from[2]) * smooth(s / c.s_lift)
        elif s < c.s_lower:
            z = peak
        else:
            z = peak + (p_to[2] - peak) * smooth((s - c.s_lower) / (1 - c.s_lower))
        p = p.copy(); p[2] = z
        return p

    # ================================================================== plan check (before simulating)
    def check_plan(self):
        """Sole margins to the step edges at every station, and swing clearance above the terrain."""
        c, terr = self.cfg, self.terrain
        problems, margins, clears = [], [], []
        for k, (sw, p_from, p_to) in enumerate(self.steps):
            heel, toe = p_to[0] + SOLE_HEEL, p_to[0] + SOLE_TOE
            x0, x1, h = self.piece_at(p_to[0] + self.SOLE_OFF[0])
            m_toe, m_heel = x1 - toe, heel - x0
            margins.append(min(m_toe, m_heel))
            if abs(p_to[2] - ANKLE_ABOVE_SOLE - h) > 1e-6 or min(m_toe, m_heel) < c.min_edge_margin:
                problems.append(f"step {k + 1}: sole margin toe {m_toe * 1e3:.0f} mm / heel {m_heel * 1e3:.0f} mm")
            # swing clearance: the sole (heel..toe) must never go into the terrain, and whenever it is
            # above a step edge (the edge lies between heel and toe) it must be min_swing_clear above it
            worst, worst_any = np.inf, np.inf
            for s in np.linspace(0.0, 1.0, 401):
                p = self.swing_pos(p_from, p_to, s)
                heel_s, toe_s = p[0] + SOLE_HEEL, p[0] + SOLE_TOE
                xs = np.linspace(heel_s, toe_s, 15)
                gap = p[2] - ANKLE_ABOVE_SOLE - max(terr.h(x) for x in xs)
                worst_any = min(worst_any, gap)
                if any(heel_s - 0.01 <= e <= toe_s + 0.01 for e in terr.edges):
                    worst = min(worst, gap)
            clears.append(worst)
            if worst < c.min_swing_clear or worst_any < -1e-3:
                problems.append(f"step {k + 1}: swing sole {worst * 1e3:.0f} mm above a step edge "
                                f"(min over the whole swing {worst_any * 1e3:.0f} mm)")
        if problems:
            raise ValueError("stairs plan rejected:\n  " + "\n  ".join(problems))
        return {"min_edge_margin": float(min(margins)), "min_swing_clear": float(min(clears))}

    # ================================================================== hooks used by the runner / logger
    def is_fallen(self):
        x, z = self.d.qpos[0], self.d.qpos[2]
        return z - self.terrain.h(x) < 0.4

    def camera_lookat(self, st):
        return [st["com"][0], 0.0, st["com"][2] - 0.2]

    video_camera = (2.6, 75, -12)

    def touchdown_info(self, foot, logger):
        """Sole margins to the edges of the tread the foot just landed on."""
        x = self.d.xpos[self.FEET[foot]][0]
        heel, toe = x + SOLE_HEEL, x + SOLE_TOE
        x0, x1, h = self.piece_at(x + self.SOLE_OFF[0])
        m_toe, m_heel = x1 - toe, heel - x0
        self.touchdowns.append((len(self.touchdowns) + 1, foot, m_toe, m_heel))
        if min(m_toe, m_heel) < 0:
            logger.event(self.d.time, "OFF_EDGE", f"{'LR'[foot]} foot sticks out over a step edge "
                         f"(toe {m_toe * 1e3:.0f} mm, heel {m_heel * 1e3:.0f} mm)")
        fmt = lambda v: "inf" if not np.isfinite(v) else f"{v * 1e3:.0f}"
        return f"tread h={h:.2f} m, toe margin {fmt(m_toe)} mm, heel margin {fmt(m_heel)} mm"

    def overlay_rows(self, st):
        x = st["com"][0]
        return [("ground h", f"{self.terrain.h(x):.2f} m (under CoM)"),
                ("CoM z / ref", f"{st['com'][2]:.3f} / {st['com_ref'][2]:.3f} m")]

    def log_extra(self):
        out = super().log_extra()
        xs, hs = self.terrain.profile()
        out.update(terrain_x=xs, terrain_h=hs, stations=np.array(self.stations),
                   z_plan=np.array([self.com_z(t) for t in self.tp]),
                   touchdowns=np.array(self.touchdowns).reshape(-1, 4),
                   rise=self.cfg.rise, tread=self.cfg.tread)
        return out
