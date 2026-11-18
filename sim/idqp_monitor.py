"""Robot state, log, events and on-screen display for the ID-QP controller (idqp_core.py).

  robot_state(ctrl, info, tau, push)  -> dict of numbers for one control step
  Logger                              -> collects states, detects events, writes npz + csv
  overlay_text(st, logger)            -> (labels, values) strings for the corner text
  add_markers(scn, ctrl, st, ...)     -> 3D markers (CoM, CoP, support polygon, push, forces)
  LiveFigures                         -> scrolling mjvFigure plots for the passive viewer
"""
from __future__ import annotations

import csv
import pathlib
from collections import deque

import mujoco
import numpy as np


# ---------------------------------------------------------------------- geometry helpers
def quat_to_rpy(q):
    """MuJoCo quaternion (w, x, y, z) -> roll, pitch, yaw (rad), ZYX convention."""
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def convex_hull(pts):
    """2D convex hull (Andrew's monotone chain), counter-clockwise, no repeated end point."""
    pts = sorted(map(tuple, pts))
    if len(pts) <= 2:
        return np.array(pts)
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def signed_margin(p, hull):
    """Distance from p to the hull boundary: > 0 inside, < 0 outside (m)."""
    if len(hull) < 3 or not np.all(np.isfinite(p)):
        return np.nan
    dmin, inside = np.inf, True
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        e = b - a
        n = np.array([e[1], -e[0]]) / np.linalg.norm(e)          # outward normal for a CCW hull
        dist = n @ (p - a)
        if dist > 0:
            inside = False
        s = np.clip((p - a) @ e / (e @ e), 0, 1)
        dmin = min(dmin, np.linalg.norm(p - (a + s * e)))
    return dmin if inside else -dmin



# ---------------------------------------------------------------------- state of one step
def robot_state(ctrl, info, tau, push):
    """All logged quantities for the current step (call right after ctrl.solve)."""
    m, d = ctrl.m, ctrl.d
    mujoco.mj_subtreeVel(m, d)
    st = dict(t=info["t"], qpos=d.qpos.copy(), qvel=d.qvel.copy(), tau=tau.copy(), push=push.copy())

    # base / torso
    st["base_pos"] = d.qpos[:3].copy()
    st["base_vel"] = d.qvel[:3].copy()
    st["pelvis_rpy"] = quat_to_rpy(d.xquat[ctrl.PELVIS])
    st["torso_rpy"] = quat_to_rpy(d.xquat[ctrl.TORSO])

    # CoM and momentum
    st["com"], st["com_ref"] = info["c"], info["c_ref"]
    st["com_vel"] = info["cd"]
    st["com_err"] = float(np.linalg.norm(info["c"] - info["c_ref"]))
    st["cdd_des"] = info["cdd_des"]
    st["cdd_qp"] = info.get("cdd_qp", np.full(3, np.nan))
    st["lin_mom"] = ctrl.mass * d.subtree_linvel[ctrl.PELVIS]
    st["ang_mom"] = d.subtree_angmom[ctrl.PELVIS].copy()          # about the whole-body CoM

    # joints: torque usage and distance to the hard joint limits
    ratio = np.abs(tau) / ctrl.tau_max
    st["tau_ratio"] = ratio
    st["tau_ratio_max"] = float(ratio.max()); st["tau_ratio_arg"] = int(ratio.argmax())
    jnt = [m.actuator_trnid[i, 0] for i in range(ctrl.nu)]
    qj = d.qpos[[m.jnt_qposadr[j] for j in jnt]]
    rng = m.jnt_range[jnt]
    dist = np.minimum(qj - rng[:, 0], rng[:, 1] - qj)
    st["jlim_dist"] = dist
    st["jlim_min"] = float(dist.min()); st["jlim_arg"] = int(dist.argmin())

    # contacts in the simulation (per foot normal force, CoP, slip ratio)
    pts = ctrl.corner_points(d)
    st["corner_pos"] = pts
    fz_sim = np.zeros(2); cop_num = np.zeros(2); slip = 0.0
    cf = np.zeros(6)
    foot_of = {b: k for k, b in enumerate(ctrl.FOOT_BODIES)}
    for k in range(d.ncon):
        con = d.contact[k]
        b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
        side = foot_of.get(b1, foot_of.get(b2))
        if side is None:
            continue
        mujoco.mj_contactForce(m, d, k, cf)
        fn = cf[0]
        if fn <= 0:
            continue
        fz_sim[side] += fn
        cop_num += fn * con.pos[:2]
        if fn > F_LOADED:      # lightly loaded points sit on the cone edge all the time; ignore them
            slip = max(slip, np.linalg.norm(cf[1:3]) / (fn * con.friction[0]))
    st["fz_sim"] = fz_sim
    st["slip_sim"] = float(slip)
    st["cop_sim"] = cop_num / fz_sim.sum() if fz_sim.sum() > 1 else np.full(2, np.nan)

    # support polygon: all corners of the feet that carry load in the simulation
    in_contact = fz_sim > 1.0
    st["foot_contact"] = in_contact
    poly_pts = [pts[i, :2] for i, (b, _) in enumerate(ctrl.corners) if in_contact[foot_of[b]]]
    hull = convex_hull(poly_pts) if len(poly_pts) >= 3 else np.zeros((0, 2))
    st["support_hull"] = hull
    st["cop_margin"] = float(signed_margin(st["cop_sim"], hull))
    st["com_margin"] = float(signed_margin(info["c"][:2], hull))

    # contact forces planned by the QP
    if "f" in info:
        f = info["f"]
        st["f_qp"] = f
        st["fz_qp"] = np.array([f[[i for i, (b, _) in enumerate(ctrl.corners) if b == fb], 2].sum()
                                for fb in ctrl.FOOT_BODIES])
        fz = f[:, 2]
        loaded = fz > F_LOADED
        fr = np.where(loaded, np.maximum(np.abs(f[:, 0]), np.abs(f[:, 1])) / (ctrl.cfg.mu * np.maximum(fz, 1e-9)), 0.0)
        st["fric_ratio"] = fr
        st["fric_ratio_max"] = float(fr.max()); st["fric_ratio_arg"] = int(fr.argmax())
        st["cop_qp"] = (pts[:, :2] * fz[:, None]).sum(0) / fz.sum() if fz.sum() > 1 else np.full(2, np.nan)
        # per foot: what decides whether a rigid foot slips (single corners may saturate harmlessly)
        ff = np.zeros(2)
        for k, fb in enumerate(ctrl.FOOT_BODIES):
            Fk = f[[i for i, (b, _) in enumerate(ctrl.corners) if b == fb]].sum(0)
            ff[k] = max(abs(Fk[0]), abs(Fk[1])) / (ctrl.cfg.mu * Fk[2]) if Fk[2] > F_LOADED else 0.0
        st["fric_foot"] = ff
        st["fric_foot_max"] = float(ff.max()); st["fric_foot_arg"] = int(ff.argmax())
    else:
        st["f_qp"] = np.full((ctrl.NC, 3), np.nan)
        st["fz_qp"] = np.full(2, np.nan)
        st["fric_ratio"] = np.full(ctrl.NC, np.nan)
        st["fric_ratio_max"] = np.nan; st["fric_ratio_arg"] = -1
        st["cop_qp"] = np.full(2, np.nan)
        st["fric_foot"] = np.full(2, np.nan)
        st["fric_foot_max"] = np.nan; st["fric_foot_arg"] = -1

    # QP solver
    for k in ("status", "status_val", "kind", "used_fallback", "iter", "prim_res", "dual_res", "solve_ms", "obj"):
        st[k] = info[k]
    tr = info.get("task_res", {})
    st["res_com"], st["res_rot"], st["res_post"] = (tr.get(k, np.nan) for k in ("com", "rot", "post"))
    act = info.get("active", {})
    st["act_tau"], st["act_fric"], st["act_lim"], st["act_fz0"] = (act.get(k, -1) for k in ("tau", "fric", "lim", "fz_zero"))
    st["active_rows"] = act.get("rows", [])
    st["diagnosis"] = info.get("diagnosis")

    # walking (walk_core.py); standing controllers leave these at their defaults
    st["phase"] = info.get("phase", "stand")
    st["phase_id"] = info.get("phase_id", -1)
    st["phase_s"] = info.get("phase_s", np.nan)
    st["contact_plan"] = info.get("contact_plan", np.array([True, True]))
    st["swing_foot"] = info.get("swing_foot", -1)
    st["zmp_ref"] = info.get("zmp_ref", np.full(2, np.nan))
    st["swing_ref"] = info.get("swing_ref", np.full(3, np.nan))
    st["swing_pos"] = info.get("swing_pos", np.full(3, np.nan))
    st["swing_err"] = float(np.linalg.norm(st["swing_ref"] - st["swing_pos"]))
    st["zmp_err"] = float(np.linalg.norm(st["cop_sim"] - st["zmp_ref"]))
    st["res_swing"] = tr.get("swing", np.nan)
    return st


# ---------------------------------------------------------------------- logger and events
ARRAY_KEYS = ["t", "qpos", "qvel", "tau", "push", "base_pos", "base_vel", "pelvis_rpy", "torso_rpy",
              "com", "com_ref", "com_vel", "com_err", "cdd_des", "cdd_qp", "lin_mom", "ang_mom",
              "tau_ratio", "tau_ratio_max", "jlim_dist", "jlim_min", "fz_sim", "slip_sim", "cop_sim",
              "foot_contact", "cop_margin", "com_margin", "f_qp", "fz_qp", "fric_ratio", "fric_ratio_max",
              "cop_qp", "fric_foot", "fric_foot_max", "status_val", "used_fallback", "iter", "prim_res", "dual_res", "solve_ms", "obj",
              "res_com", "res_rot", "res_post", "act_tau", "act_fric", "act_lim", "act_fz0",
              "phase_id", "phase_s", "contact_plan", "swing_foot", "zmp_ref", "swing_ref", "swing_pos",
              "swing_err", "zmp_err", "res_swing"]
PHASE_NAMES = {0: "stand", 1: "double support", 2: "single support", 3: "stand (end)"}
DEBOUNCE = 5           # control steps a condition must last before it is reported (10 ms):
                       # contact mismatches, friction limit, slip (brief touches during load transfer are normal)

F_LOADED = 10.0        # N: corners / contact points below this are ignored for friction and slip ratios
# (on, off) thresholds: an event starts when the value crosses `on` and ends only after it crosses `off`
THRESH = dict(tau=(0.98, 0.90), fric=(0.98, 0.90), slip=(0.95, 0.85), margin=(0.010, 0.015), jlim=(0.02, 0.03))


def _nanmin(v):
    """nanmin that returns nan (no warning) when nothing is finite, e.g. a run stopped at t = 0."""
    v = np.asarray(v, dtype=float)
    return float(np.nanmin(v)) if np.isfinite(v).any() else np.nan


class Logger:
    def __init__(self, ctrl, out_dir, verbose=True, prefix="idqp"):
        self.ctrl, self.out, self.prefix = ctrl, pathlib.Path(out_dir), prefix
        self.out.mkdir(parents=True, exist_ok=True)
        self.rows = {k: [] for k in ARRAY_KEYS}
        self.status = []
        self.events = []            # (t, type, detail)
        self.flags = {}
        self.verbose = verbose
        self.n_infeasible = 0; self.n_warn = 0; self.n_fallback = 0
        self.last_diag = None; self.last_infeasible_t = -np.inf
        self.streak = 0
        self.count = {}            # debounce counters
        self.prev = None           # previous state (for phase transitions)
        self.n_steps_done = 0

    def event(self, t, kind, detail=""):
        self.events.append((round(t, 4), kind, detail))
        if self.verbose:
            print(f"[{t:6.3f}s] {kind:<16} {detail}", flush=True)

    def _edge(self, key, on, off, t, kind_on, detail_on, kind_off=None, detail_off=""):
        """Event when `on` becomes true; the matching end event only once `off` is true (hysteresis)."""
        was = self.flags.get(key, False)
        if not was and on:
            self.event(t, kind_on, detail_on); self.flags[key] = True
        elif was and off:
            if kind_off:
                self.event(t, kind_off, detail_off)
            self.flags[key] = False

    def add(self, st):
        for k in ARRAY_KEYS:
            self.rows[k].append(st[k])
        self.status.append(st["status"])
        t, c = st["t"], self.ctrl

        # QP status: log when the status changes (not every step), plus when the infeasibility cause changes
        kind, prev = st["kind"], self.flags.get("qp_kind", "ok")
        if kind == "infeasible":
            self.n_infeasible += 1; self.last_infeasible_t = t
            diag = st["diagnosis"]
            cause = diag["text"] if diag else ""
            # new event only when infeasibility starts or its main groups change (shares drift every step)
            if prev != "infeasible" or (diag or {}).get("main") != (self.last_diag or {}).get("main"):
                top = ", ".join(n for n, _ in diag["top"][:3]) if diag else ""
                self.event(t, "QP_INFEASIBLE", f"{st['status']} | {cause} | top rows: {top} | holding previous tau")
            self.last_diag = diag
        elif kind == "warn":
            self.n_warn += 1
            if prev != "warn":
                self.event(t, "QP_WARN", f"{st['status']} iter={st['iter']} prim_res={st['prim_res']:.1e}")
        elif kind == "error":
            self.event(t, "QP_ERROR", st["status"])
        elif prev != "ok":
            self.event(t, "QP_OK", f"solved again after {self.streak} non-solved step(s)")
        self.streak = self.streak + 1 if kind != "ok" else 0
        self.flags["qp_kind"] = kind
        self.n_fallback += int(st["used_fallback"])

        T = THRESH
        r = st["tau_ratio_max"]
        self._edge("tau", r >= T["tau"][0], r < T["tau"][1], t, "TAU_SATURATED",
                   f"{c.act_names[st['tau_ratio_arg']]} |tau|/tau_max={r:.2f}", "TAU_OK")
        fr = st["fric_foot_max"] if np.isfinite(st["fric_foot_max"]) else 0.0
        self._edge("fric", self._debounced("fric", fr >= T["fric"][0]), fr < T["fric"][1], t, "FRICTION_LIMIT",
                   f"{'LR'[st['fric_foot_arg']]} foot |sum f_t|/(mu sum f_z)={fr:.2f} (QP)", "FRICTION_OK")
        s_ = st["slip_sim"]
        self._edge("slip", self._debounced("slip", s_ >= T["slip"][0]), s_ < T["slip"][1], t, "SLIP_SIM", f"|f_t|/(mu f_n)={s_:.2f}", "SLIP_OK")
        mg = st["cop_margin"]
        if np.isfinite(mg) and t > 0.01:
            self._edge("margin", mg < T["margin"][0], mg > T["margin"][1], t, "COP_NEAR_EDGE",
                       f"CoP margin={mg * 1e3:.1f} mm", "COP_OK")
        jl = st["jlim_min"]
        self._edge("jlim", jl < T["jlim"][0], jl > T["jlim"][1], t, "JOINT_NEAR_LIMIT",
                   f"{c.act_names[st['jlim_arg']]} dist={jl:.3f} rad", "JOINT_OK")
        if st["phase_id"] < 0:                       # standing controller: any lift is news
            for k, side in enumerate("LR"):
                lifted = not st["foot_contact"][k]
                self._edge(f"lift{side}", lifted, not lifted, t, "FOOT_LIFT", f"{side} foot fz_sim<1 N", "FOOT_TOUCH", f"{side} foot")
        else:
            self._walk_events(st, t)
        self.prev = st
        pushing = bool(np.any(st["push"]))
        self._edge("push", pushing, not pushing, t, "PUSH_START", f"F={np.round(st['push'], 1)} N", "PUSH_END")

    def _debounced(self, key, cond):
        self.count[key] = self.count.get(key, 0) + 1 if cond else 0
        return self.count[key] >= DEBOUNCE

    def _walk_events(self, st, t):
        """Walking: compare the real contacts with the planned ones, and report phase transitions."""
        prev = self.prev
        if prev is not None and prev["phase_id"] != st["phase_id"] or prev is None:
            if prev is not None and prev["phase_id"] == 2:          # end of a swing: touchdown error
                self.n_steps_done += 1
                extra = ""
                if hasattr(self.ctrl, "touchdown_info"):              # optional controller hook (stairs)
                    extra = " | " + self.ctrl.touchdown_info(prev["swing_foot"], self)
                self.event(t, "TOUCHDOWN", f"step {self.n_steps_done}: {'LR'[prev['swing_foot']]} foot, "
                           f"error to target {prev['swing_err'] * 1e3:.1f} mm" + extra)
            detail = PHASE_NAMES[st["phase_id"]]
            if st["phase_id"] == 2:
                detail += f", swing {'LR'[st['swing_foot']]}"
            self.event(t, "PHASE", detail)
        s_ = st["phase_s"]
        for k, side in enumerate("LR"):
            planned, real = st["contact_plan"][k], st["foot_contact"][k]
            # stance foot that carries no load (foot tipping or bouncing)
            self._edge(f"slift{side}", self._debounced(f"slift{side}", planned and not real), planned and real or not planned,
                       t, "STANCE_LIFT", f"{side} foot should be in contact but fz_sim<1 N", "STANCE_OK", f"{side} foot")
            # swing foot touching the floor in the middle of the swing (scuffing / early touchdown)
            mid = np.isfinite(s_) and 0.15 < s_ < 0.85
            self._edge(f"scuff{side}", self._debounced(f"scuff{side}", (not planned) and real and mid), (not real) or planned,
                       t, "SWING_SCUFF", f"{side} foot touches the floor at s={s_:.2f} of the swing", "SWING_CLEAR", f"{side} foot")

    def save(self, tag=""):
        arrays = {}
        for k, v in self.rows.items():
            if v:
                try:
                    arrays[k] = np.array(v)
                except ValueError:
                    arrays[k] = np.array(v, dtype=object)
        c = self.ctrl
        npz = self.out / f"{self.prefix}_log{tag}.npz"
        extra = c.log_extra() if hasattr(c, "log_extra") else {}
        np.savez(npz, **arrays, **extra, status=np.array(self.status), act_names=np.array(c.act_names),
                 corner_names=np.array(c.corner_names), tau_max=c.tau_max, mu=c.cfg.mu,
                 tau_scale=c.cfg.tau_scale, push_scale=c.cfg.push_scale,
                 pushes=np.array([[p[0], p[1], *(p[2] * c.cfg.push_scale)] for p in c.cfg.pushes]))
        ev = self.out / f"{self.prefix}_events{tag}.csv"
        with open(ev, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["t", "event", "detail"]); w.writerows(self.events)
        return npz, ev

    def summary(self):
        n = len(self.status)
        if not n:
            return "no steps"
        r = self.rows
        return (f"{n} steps, t_end={r['t'][-1]:.2f}s | infeasible {self.n_infeasible}, inaccurate/maxiter {self.n_warn}, "
                f"fallback {self.n_fallback} | max CoM err {np.max(r['com_err']) * 1e3:.1f} mm | "
                f"max tau ratio {np.max(r['tau_ratio_max']):.2f} | max foot friction use {np.nanmax(r['fric_foot_max']):.2f} | "
                f"min CoP margin {_nanmin(r['cop_margin']) * 1e3:.1f} mm | QP mean {np.mean(r['solve_ms']):.2f} ms")


# ---------------------------------------------------------------------- display
def overlay_text(st, logger):
    """Two columns of text for the viewer corner / video frame."""
    c = logger.ctrl
    status = st["status"].upper() if st["kind"] != "ok" else st["status"]
    if st["used_fallback"]:
        status += "  (HOLD tau)"
    fr = st["fric_foot_max"]
    walk = st["phase_id"] >= 0
    rows = [
        ("t", f"{st['t']:.2f} s"),
        ("QP", status),
        ("iter / solve", f"{st['iter']} / {st['solve_ms']:.1f} ms"),
        ("CoM err", f"{st['com_err'] * 1e3:.1f} mm"),
        ("|L| about CoM", f"{np.linalg.norm(st['ang_mom']):.2f} N m s"),
        ("tau/tau_max", f"{st['tau_ratio_max']:.2f} ({c.act_names[st['tau_ratio_arg']]})"),
        ("friction (QP)", f"{fr:.2f} ({'LR'[st['fric_foot_arg']]} foot)" if np.isfinite(fr) else "-"),
        ("slip (sim)", f"{st['slip_sim']:.2f}"),
        ("CoP margin", f"{st['cop_margin'] * 1e3:.1f} mm"),
        ("fz L / R", f"{st['fz_sim'][0]:.0f} / {st['fz_sim'][1]:.0f} N"),
        ("push", f"{np.round(st['push'], 0)} N" if np.any(st["push"]) else "-"),
        ("infeasible", f"{logger.n_infeasible} steps"),
    ]
    if walk:
        ph = PHASE_NAMES[st["phase_id"]] + (f", swing {'LR'[st['swing_foot']]}" if st["phase_id"] == 2 else "")
        rows[1:1] = [("phase", ph), ("steps done", f"{logger.n_steps_done} / {len(getattr(c, 'steps', []))}"),
                     ("ZMP err", f"{st['zmp_err'] * 1e3:.1f} mm" if np.isfinite(st["zmp_err"]) else "-"),
                     ("swing err", f"{st['swing_err'] * 1e3:.1f} mm" if np.isfinite(st["swing_err"]) else "-"),
                     ("CoM x", f"{st['com'][0]:.3f} m")]
    if hasattr(c, "overlay_rows"):                     # optional controller hook (stairs)
        rows[6:6] = c.overlay_rows(st)
    if logger.last_diag:
        rows.append(("last cause", f"t={logger.last_infeasible_t:.2f}s {logger.last_diag['text']}"))
        for name, _ in logger.last_diag["top"][:2]:
            rows.append(("", name))
    return "\n".join(r[0] for r in rows), "\n".join(r[1] for r in rows)


def _add_geom(scn, gtype, size, pos, rgba, mat=np.eye(3)):
    if scn.ngeom >= scn.maxgeom:
        return None
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float), mat.flatten(), np.asarray(rgba, np.float32))
    scn.ngeom += 1
    return g


def _add_line(scn, a, b, rgba, width=0.004, gtype=mujoco.mjtGeom.mjGEOM_CAPSULE):
    g = _add_geom(scn, gtype, [width, 0, 0], np.zeros(3), rgba)
    if g is not None:
        mujoco.mjv_connector(g, gtype, width, np.asarray(a, float), np.asarray(b, float))


def add_markers(scn, ctrl, st, recent_infeasible=False, force_scale=0.002):
    """Append CoM, CoP, support polygon, push arrow and QP contact forces to an mjvScene."""
    z = 0.003
    hull = st["support_hull"]
    if recent_infeasible:
        poly_rgba = [1, 0.1, 0.1, 1]
    elif st["cop_margin"] < THRESH["margin"][0]:
        poly_rgba = [1, 0.6, 0, 1]
    else:
        poly_rgba = [0.1, 0.8, 0.2, 1]
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        _add_line(scn, [a[0], a[1], z], [b[0], b[1], z], poly_rgba, 0.005)
    c = st["com"]
    _add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.02, 0, 0], [c[0], c[1], z], [0.2, 0.4, 1, 1])        # CoM on floor
    _add_line(scn, [c[0], c[1], z], c, [0.2, 0.4, 1, 0.4], 0.002)
    cr = st["com_ref"]
    _add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.012, 0, 0], [cr[0], cr[1], z], [0.6, 0.8, 1, 0.8])   # CoM ref
    if np.all(np.isfinite(st["cop_sim"])):
        p = st["cop_sim"]
        _add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [0.015, 0.015, 0.004], [p[0], p[1], z], [1, 1, 0, 1])     # CoP sim
    if np.all(np.isfinite(st["f_qp"])):
        for pw, f in zip(st["corner_pos"], st["f_qp"]):
            if f[2] > 1:
                _add_line(scn, pw, pw + force_scale * f, [1, 0.5, 0, 0.9], 0.003)                         # QP forces
    if np.all(np.isfinite(st["zmp_ref"])):
        p = st["zmp_ref"]
        _add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.012, 0, 0], [p[0], p[1], z], [1, 0, 1, 1])        # ZMP ref
    if np.all(np.isfinite(st["swing_ref"])):
        _add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.015, 0, 0], st["swing_ref"], [0, 1, 1, 0.8])      # swing target
    for fp in getattr(ctrl, "footsteps", []):
        cx, cy = fp[0] + ctrl.SOLE_OFF[0], fp[1] + ctrl.SOLE_OFF[1]
        _add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [0.07, 0.03, 0.001], [cx, cy, 0.001], [0, 0.8, 0.8, 0.25])  # planned steps
    F = st["push"]
    if np.any(F):
        tc = ctrl.d.xipos[ctrl.TORSO]
        _add_line(scn, tc - 0.006 * F, tc, [1, 0, 0, 1], 0.012, mujoco.mjtGeom.mjGEOM_ARROW)


class LiveFigures:
    """Scrolling plots in the passive viewer (mjvFigure), last `window` seconds."""

    SPECS = [("CoM err (mm)", lambda s: s["com_err"] * 1e3, None),
             ("tau / tau_max", lambda s: s["tau_ratio_max"], (0, 1.1)),
             ("friction use per foot (QP)", lambda s: s["fric_foot_max"] if np.isfinite(s["fric_foot_max"]) else 0, (0, 1.1))]

    def __init__(self, window=5.0, rate=20.0):
        self.window = window
        self.buf = [deque(maxlen=int(window * rate) + 1) for _ in self.SPECS]
        self.tbuf = deque(maxlen=int(window * rate) + 1)
        self.figs = []
        for title, _, yr in self.SPECS:
            f = mujoco.MjvFigure()
            mujoco.mjv_defaultFigure(f)
            f.title = title
            f.flg_extend = 1
            f.gridsize[:] = [3, 3]
            f.figurergba[:] = [0, 0, 0, 0.5]
            f.linergb[0] = [1, 0.8, 0.2]
            f.linewidth = 2
            if yr:
                f.range[1] = yr
            self.figs.append(f)

    def push(self, st):
        self.tbuf.append(st["t"])
        for b, (_, fn, _) in zip(self.buf, self.SPECS):
            b.append(fn(st))

    def draw(self, viewer):
        vp = viewer.viewport
        if vp is None:
            return
        w, h = vp.width // len(self.figs), vp.height // 4
        t = np.array(self.tbuf)
        out = []
        for i, (f, b, (_, _, yr)) in enumerate(zip(self.figs, self.buf, self.SPECS)):
            y = np.array(b); n = len(y)
            f.linepnt[0] = n
            f.linedata[0][0:2 * n:2] = t
            f.linedata[0][1:2 * n:2] = y
            if n:
                f.range[0] = [t[-1] - self.window, t[-1]]
                if yr is None:
                    f.range[1] = [0, max(5.0, float(y.max()) * 1.2)]
            out.append((mujoco.MjrRect(i * w, 0, w, h), f))
        viewer.set_figures(out)
