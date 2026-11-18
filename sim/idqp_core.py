"""ID-QP balance controller for the G1, same formulation as g1_demo.py, packaged as a class.

Differences from g1_demo.py (which stays untouched):
  * solve() returns a full info dict: OSQP status/iterations/residuals, constraint matrix and
    bounds, row groups and row names, per-task residuals, active constraints.
  * Infeasible / failed solves do not stop the run: the previous torque is held (fallback)
    and the primal infeasibility certificate is turned into a diagnosis (which constraints conflict).
  * mj_forward is called at the start of every control step, so M, h, Jacobians and contacts
    all match the current (q, qd).  g1_demo.py uses the values left over from the previous mj_step.
  * Scenario knobs: friction coefficient, torque-limit scale, push scale.

Decision vector x = [qdd (nv) | tau (nu) | f (3 per foot corner)].
Constraint rows, in order: dyn (nv) | feet (12) | tau (nu) | fric (5 per corner) | lim (nu).
"""
from __future__ import annotations

import os
import pathlib
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np
import osqp
import scipy.sparse as sp

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_XML = ROOT / "third_party/mujoco_menagerie/unitree_g1/scene.xml"

S_ = osqp.SolverStatus
OK = {int(S_.OSQP_SOLVED)}
WARN = {int(S_.OSQP_SOLVED_INACCURATE), int(S_.OSQP_MAX_ITER_REACHED), int(S_.OSQP_TIME_LIMIT_REACHED)}
INFEASIBLE = {int(S_.OSQP_PRIMAL_INFEASIBLE), int(S_.OSQP_PRIMAL_INFEASIBLE_INACCURATE)}
# anything else (dual infeasible, non convex, unsolved) means the QP itself is broken -> stop


@dataclass
class Config:
    xml: str = os.environ.get("G1_XML", str(DEFAULT_XML))
    mu: float = 0.5                 # friction coefficient used in the QP (MuJoCo floor uses 1.0)
    tau_scale: float = 1.0          # scales actuator torque limits (QP bound and ctrlrange)
    push_scale: float = 1.0         # scales the scripted pushes
    # gains and weights, same as g1_demo.py
    kp_com: float = 100.0; kd_com: float = 20.0
    kp_rot: float = 200.0; kd_rot: float = 30.0
    kp_post: float = 60.0; kd_post: float = 15.0
    kf: float = 20.0
    w_com: float = 100.0; w_rot: float = 20.0; w_post: float = 1.0; w_post_leg: float = 0.05
    w_tau: float = 1e-4; w_f: float = 1e-5; w_qdd: float = 1e-6
    fz_max: float = 600.0
    h_lim: float = 0.08
    knee_min: float = 0.25
    # pushes: (t_start, duration, force_world), applied at the torso COM
    pushes: list = field(default_factory=lambda: [(6.5, 0.2, np.array([60.0, 0, 0])),
                                                  (8.0, 0.2, np.array([0, -60.0, 0]))])


def smooth(s):
    """0 -> 1 quintic with zero velocity and acceleration at both ends."""
    s = np.clip(s, 0, 1)
    return s * s * s * (10 - 15 * s + 6 * s * s)


class IDQPController:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg = cfg or Config()
        m = self.m = mujoco.MjModel.from_xml_path(cfg.xml)
        self.d = mujoco.MjData(m)
        self.d2 = mujoco.MjData(m)            # scratch data for dJ*qd by finite difference
        self.nv, self.nu = m.nv, m.nu
        nv, nu = self.nv, self.nu

        # position actuators -> pure torque motors: tau = ctrl
        self.tau_max = m.actuator_forcerange[:, 1].copy() * cfg.tau_scale
        m.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
        m.actuator_gainprm[:] = 0; m.actuator_gainprm[:, 0] = 1.0
        m.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_NONE
        m.actuator_biasprm[:] = 0
        m.actuator_ctrllimited[:] = 1
        m.actuator_ctrlrange[:, 0] = -self.tau_max; m.actuator_ctrlrange[:, 1] = self.tau_max
        self.act_dof = np.array([m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in range(nu)])
        self.S = np.zeros((nu, nv)); self.S[np.arange(nu), self.act_dof] = 1.0
        self.act_names = [m.actuator(i).name.replace("_joint", "") for i in range(nu)]

        # nominal pose: knees slightly bent, feet just touching the floor
        jq = lambda n: m.joint(n).qposadr[0]
        q = m.key_qpos[0].copy()
        for side in ["left", "right"]:
            q[jq(f"{side}_hip_pitch_joint")] = -0.312
            q[jq(f"{side}_knee_joint")] = 0.669
            q[jq(f"{side}_ankle_pitch_joint")] = -0.357
        d = self.d
        d.qpos[:] = q; mujoco.mj_kinematics(m, d)
        self.foot_geoms = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
                           and m.body(m.geom_bodyid[g]).name.endswith("ankle_roll_link")]
        assert len(self.foot_geoms) == 8, f"expected 8 foot spheres, got {len(self.foot_geoms)}"
        zmin = min(d.geom_xpos[g][2] - m.geom_size[g, 0] for g in self.foot_geoms)
        q[2] -= zmin - 1e-4
        self.q_nom = q

        self.PELVIS = m.body("pelvis").id
        self.TORSO = m.body("torso_link").id
        self.FOOT_BODIES = [m.body("left_ankle_roll_link").id, m.body("right_ankle_roll_link").id]
        self.corners = []                     # (body, local contact point)
        for g in self.foot_geoms:
            p = m.geom_pos[g].copy(); p[2] -= m.geom_size[g, 0]
            self.corners.append((int(m.geom_bodyid[g]), p))
        self.NC = len(self.corners); self.nf = 3 * self.NC
        self.NX = nv + nu + self.nf
        self.corner_names = []
        count = {b: 0 for b in self.FOOT_BODIES}
        for b, _ in self.corners:
            self.corner_names.append(f"{'L' if b == self.FOOT_BODIES[0] else 'R'}c{count[b]}")
            count[b] += 1
        self.iq = slice(0, nv); self.it = slice(nv, nv + nu); self.iff = slice(nv + nu, self.NX)
        self.mass = float(m.body_subtreemass[self.PELVIS])

        self._build_constant_constraints()
        self._build_row_names()
        self._jp = np.zeros((3, nv)); self._jr = np.zeros((3, nv))
        self.tau_prev = np.zeros(nu)
        self.reset()

    # ------------------------------------------------------------------ setup helpers
    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:] = self.q_nom
        mujoco.mj_forward(self.m, self.d)
        self.c0 = self.d.subtree_com[self.PELVIS].copy()
        self.tau_prev = np.zeros(self.nu)

    def _build_constant_constraints(self):
        m, cfg, nv, nu, NX = self.m, self.cfg, self.nv, self.nu, self.NX
        rows = []
        for i in range(self.NC):
            fx, fy, fz = nv + nu + 3 * i, nv + nu + 3 * i + 1, nv + nu + 3 * i + 2
            for a, s in [(fx, 1), (fx, -1), (fy, 1), (fy, -1)]:
                r = np.zeros(NX); r[a] = s; r[fz] = -cfg.mu; rows.append(r)   # +-f_t - mu f_z <= 0
            r = np.zeros(NX); r[fz] = 1; rows.append(r)                        # 0 <= f_z <= fz_max
        self.A_fric = np.array(rows)
        self.l_fric = np.tile([-np.inf] * 4 + [0.0], self.NC)
        self.u_fric = np.tile([0.0] * 4 + [cfg.fz_max], self.NC)
        self.A_tau = np.zeros((nu, NX)); self.A_tau[:, self.it] = np.eye(nu)
        jnt = [m.actuator_trnid[i, 0] for i in range(nu)]
        self.jlo = np.array([m.jnt_range[j, 0] for j in jnt]) + 0.05
        self.jhi = np.array([m.jnt_range[j, 1] for j in jnt]) - 0.05
        for side in ["left", "right"]:
            self.jlo[m.actuator(f"{side}_knee_joint").id] = cfg.knee_min
        self.A_lim = np.zeros((nu, NX)); self.A_lim[np.arange(nu), self.act_dof] = 1.0

    def _build_row_names(self):
        m, nv, nu, NC = self.m, self.nv, self.nu, self.NC
        dof_names = []
        for j in range(nv):
            jid = m.dof_jntid[j]
            if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                dof_names.append(["base_x", "base_y", "base_z", "base_wx", "base_wy", "base_wz"][j - m.jnt_dofadr[jid]])
            else:
                dof_names.append(m.joint(jid).name.replace("_joint", ""))
        names = [f"dyn[{n}]" for n in dof_names]
        names += [f"feet[{s}.{k}]" for s in "LR" for k in ["vx", "vy", "vz", "wx", "wy", "wz"]]
        names += [f"tau[{n}]" for n in self.act_names]
        names += [f"fric[{c} {k}]" for c in self.corner_names for k in ["+fx", "-fx", "+fy", "-fy", "fz"]]
        names += [f"lim[{n}]" for n in self.act_names]
        self.row_names = names
        sizes = [("dyn", nv), ("feet", 12), ("tau", nu), ("fric", 5 * NC), ("lim", nu)]
        self.groups, k = {}, 0
        for g, n in sizes:
            self.groups[g] = slice(k, k + n); k += n
        self.n_rows = k

    # ------------------------------------------------------------------ references
    def com_ref_pos(self, t):
        c = self.c0.copy()
        c[1] += 0.05 * (smooth((t - 1.0) / 1.5) - smooth((t - 3.0) / 1.0))   # 5 cm left, then back
        c[2] -= 0.06 * (smooth((t - 4.0) / 1.0) - smooth((t - 5.0) / 1.0))   # squat 6 cm, then up
        return c

    def com_ref(self, t, h=1e-3):
        p, pp, pm = self.com_ref_pos(t), self.com_ref_pos(t + h), self.com_ref_pos(t - h)
        return p, (pp - pm) / (2 * h), (pp - 2 * p + pm) / (h * h)

    def push_force(self, t):
        for t0, T, F in self.cfg.pushes:
            if t0 <= t < t0 + T:
                return F * self.cfg.push_scale
        return np.zeros(3)

    # ------------------------------------------------------------------ jacobians
    def task_jacobians(self, data):
        m, jp, jr = self.m, self._jp, self._jr
        Jf = []
        for b in self.FOOT_BODIES:
            mujoco.mj_jacBody(m, data, jp, jr, b)
            Jf += [jp.copy(), jr.copy()]
        Jcom = np.zeros((3, self.nv)); mujoco.mj_jacSubtreeCom(m, data, Jcom, self.PELVIS)
        mujoco.mj_jacBody(m, data, jp, jr, self.TORSO)
        return np.vstack(Jf), Jcom, jr.copy()

    def dJqd(self, q, qd, eps=1e-6):
        m, d2 = self.m, self.d2
        d2.qpos[:] = q; mujoco.mj_kinematics(m, d2); mujoco.mj_comPos(m, d2)
        J0 = self.task_jacobians(d2)
        d2.qpos[:] = q; mujoco.mj_integratePos(m, d2.qpos, qd, eps)
        mujoco.mj_kinematics(m, d2); mujoco.mj_comPos(m, d2)
        J1 = self.task_jacobians(d2)
        return [((b - a) / eps) @ qd for a, b in zip(J0, J1)]

    def corner_points(self, data):
        return np.array([data.xpos[b] + data.xmat[b].reshape(3, 3) @ p for b, p in self.corners])

    def corner_jac(self, data):
        Jc = np.zeros((self.nf, self.nv))
        for i, ((b, _), pw) in enumerate(zip(self.corners, self.corner_points(data))):
            mujoco.mj_jac(self.m, data, self._jp, None, pw, b)
            Jc[3 * i:3 * i + 3] = self._jp
        return Jc

    # ------------------------------------------------------------------ the QP
    def solve(self, t):
        """Build and solve the ID-QP at the current state. Returns (tau_to_apply, info)."""
        m, d, cfg = self.m, self.d, self.cfg
        nv, nu, NX, iq, it, iff = self.nv, self.nu, self.NX, self.iq, self.it, self.iff
        mujoco.mj_forward(m, d)
        q, qd = d.qpos.copy(), d.qvel.copy()
        M = np.zeros((nv, nv)); mujoco.mj_fullM(m, d, M)
        h = d.qfrc_bias.copy()
        Jf, Jcom, Jrot = self.task_jacobians(d)
        dJf, dJcom, dJrot = self.dJqd(q, qd)
        Jc = self.corner_jac(d)

        # tasks ||A x - b||^2
        tasks = {}
        c, cd = d.subtree_com[self.PELVIS].copy(), Jcom @ qd
        cr, cdr, cddr = self.com_ref(t)
        a_com = cddr + cfg.kp_com * (cr - c) + cfg.kd_com * (cdr - cd)
        A = np.zeros((3, NX)); A[:, iq] = Jcom; tasks["com"] = (cfg.w_com, A, a_com - dJcom)

        err = np.zeros(3)
        mujoco.mju_subQuat(err, np.array([1.0, 0, 0, 0]), d.xquat[self.TORSO])
        err = d.xmat[self.TORSO].reshape(3, 3) @ err
        a_rot = cfg.kp_rot * err - cfg.kd_rot * (Jrot @ qd)
        A = np.zeros((3, NX)); A[:, iq] = Jrot; tasks["rot"] = (cfg.w_rot, A, a_rot - dJrot)

        a_post = cfg.kp_post * (self.q_nom[7:] - q[7:]) - cfg.kd_post * qd[6:]
        wj = np.sqrt(np.where(np.arange(nu) < 12, cfg.w_post_leg, cfg.w_post))
        A = np.zeros((nv - 6, NX)); A[:, 6:nv] = np.diag(wj); tasks["post"] = (1.0, A, wj * a_post)

        P = np.zeros((NX, NX)); g = np.zeros(NX)
        for w, A, b in tasks.values():
            P += 2 * w * A.T @ A; g -= 2 * w * A.T @ b
        P[iq, iq] += 2 * cfg.w_qdd * np.eye(nv)
        P[it, it] += 2 * cfg.w_tau * np.eye(nu)
        P[iff, iff] += 2 * cfg.w_f * np.eye(self.nf)

        # constraints
        A_dyn = np.hstack([M, -self.S.T, -Jc.T]); b_dyn = -h
        A_feet = np.hstack([Jf, np.zeros((12, nu + self.nf))]); b_feet = -dJf - cfg.kf * (Jf @ qd)
        qj, vj = q[7:], qd[6:]
        qdd_lo = 2 * (self.jlo - qj - vj * cfg.h_lim) / cfg.h_lim ** 2
        qdd_hi = 2 * (self.jhi - qj - vj * cfg.h_lim) / cfg.h_lim ** 2
        qdd_lo = np.minimum(qdd_lo, qdd_hi)
        Acons = np.vstack([A_dyn, A_feet, self.A_tau, self.A_fric, self.A_lim])
        lo = np.concatenate([b_dyn, b_feet, -self.tau_max, self.l_fric, qdd_lo])
        hi = np.concatenate([b_dyn, b_feet, self.tau_max, self.u_fric, np.maximum(qdd_hi, qdd_lo)])

        base = dict(t=t, c=c, cd=cd, c_ref=cr, cdd_des=a_com, a_rot=a_rot, rot_err=err, Jcom=Jcom, dJcom=dJcom, Jc=Jc)
        return self.solve_qp(P, g, Acons, lo, hi, tasks, base)

    def solve_qp(self, P, g, Acons, lo, hi, tasks, info, groups=None, row_names=None):
        """Solve with OSQP, apply the fallback rule, and add solver info to `info`.

        Shared by every controller built on this class. `groups` / `row_names` describe the rows of
        Acons when they differ from the standing layout (e.g. walking: one foot in contact).
        """
        groups = groups or self.groups
        row_names = row_names or self.row_names
        it, iq, iff = self.it, self.iq, self.iff
        solver = osqp.OSQP()
        solver.setup(sp.csc_matrix(np.triu(P)), g, sp.csc_matrix(Acons), lo, hi, verbose=False,
                     eps_abs=1e-6, eps_rel=1e-6, polishing=True, max_iter=20000)
        t0 = time.perf_counter()
        res = solver.solve()
        solve_ms = (time.perf_counter() - t0) * 1e3

        sv = int(res.info.status_val)
        kind = "ok" if sv in OK else "warn" if sv in WARN else "infeasible" if sv in INFEASIBLE else "error"
        x = res.x
        usable = kind in ("ok", "warn") and x is not None and np.all(np.isfinite(x))
        if usable:
            tau = x[it].copy()
            self.tau_prev = tau
        else:
            tau = self.tau_prev.copy()        # fallback: hold the previous torque

        info.update(status=res.info.status, status_val=sv, kind=kind, used_fallback=not usable,
                    iter=int(res.info.iter), prim_res=float(res.info.prim_res), dual_res=float(res.info.dual_res),
                    obj=float(res.info.obj_val), solve_ms=solve_ms)
        if usable:
            info["x"] = x
            info["qdd"], info["tau_qp"], info["f"] = x[iq], x[it], x[iff].reshape(self.NC, 3)
            info["cdd_qp"] = info["Jcom"] @ x[iq] + info["dJcom"]
            info["task_res"] = {k: float(np.linalg.norm(A @ x - b)) for k, (w, A, b) in tasks.items()}
            info["active"] = self._active_counts(Acons @ x, lo, hi, groups, row_names)
        if kind == "infeasible" and res.prim_inf_cert is not None:
            info["diagnosis"] = self.diagnose(np.asarray(res.prim_inf_cert), groups=groups, row_names=row_names)
        return tau, info

    # ------------------------------------------------------------------ analysis helpers
    def _active_counts(self, Ax, lo, hi, groups=None, row_names=None):
        """Number of inequality rows sitting on a bound, per group (dyn/feet are equalities)."""
        groups = groups or self.groups
        row_names = row_names or self.row_names
        tol = 1e-4 * np.maximum(1.0, np.abs(np.where(np.isfinite(lo), lo, 0)))
        on_lo = np.isfinite(lo) & (Ax - lo < tol)
        tol = 1e-4 * np.maximum(1.0, np.abs(np.where(np.isfinite(hi), hi, 0)))
        on_hi = np.isfinite(hi) & (hi - Ax < tol)
        act = on_lo | on_hi
        out = {}
        for gname in ("tau", "fric", "lim"):
            out[gname] = int(act[groups[gname]].sum())
        # friction: f_z = 0 rows mean a corner is unloaded, not a friction limit; count them separately
        fz_rows = np.arange(groups["fric"].start + 4, groups["fric"].stop, 5)
        out["fz_zero"] = int(on_lo[fz_rows].sum())
        out["fric"] -= out["fz_zero"]
        fz_set = set(fz_rows.tolist())
        out["rows"] = [row_names[i] for i in np.where(act)[0] if i not in fz_set and i >= groups["tau"].start]
        return out

    def diagnose(self, y, top=6, groups=None, row_names=None):
        """Turn the primal infeasibility certificate y into 'which constraints conflict'.

        OSQP returns y with A^T y ~ 0 and a negative support value: rows with a large |y|
        are the ones that cannot be satisfied together.
        """
        groups = groups or self.groups
        row_names = row_names or self.row_names
        a = np.abs(y)
        total = a.sum()
        if total == 0 or not np.isfinite(total):
            return {"share": {}, "top": [], "text": "empty certificate", "main": ()}
        share = {gname: float(a[sl].sum() / total) for gname, sl in groups.items() if sl.stop > sl.start}
        idx = np.argsort(-a)[:top]
        top_rows = [(row_names[i], float(a[i] / a.max())) for i in idx if a[i] > 1e-3 * a.max()]
        text = " ".join(f"{k} {v * 100:.0f}%" for k, v in sorted(share.items(), key=lambda kv: -kv[1]) if v >= 0.05)
        main = tuple(sorted(k for k, v in share.items() if v >= 0.15))   # groups that carry the conflict
        return {"share": share, "top": top_rows, "text": text, "main": main}
