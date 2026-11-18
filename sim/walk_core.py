"""G1 walking: footstep plan -> ZMP reference -> ZMP preview control (Kajita 2003, LIPM)
-> CoM trajectory + swing-foot trajectory -> ID-QP whose contact set changes with the phase.

Same algorithm as g1_walk_demo.py, built on IDQPController (idqp_core.py) so it gets the same
solver info, infeasibility diagnosis, logging and viewer as the balance controller.

Differences from g1_walk_demo.py:
  * mj_forward at the start of every control step (M, h, J, contacts match the current q, qd).
  * Infeasible QP: hold the previous torque and diagnose which constraint rows conflict.
  * The constraint rows change with the phase (6 foot rows per stance foot); row names and groups
    are rebuilt every step so the diagnosis names the right rows.

Decision vector x = [qdd (nv) | tau (nu) | f (3 per foot corner)].
Tasks: CoM (LIPM plan), torso and pelvis orientation, swing foot 6D (single support), posture.
Constraints: dynamics, stance feet fixed (6D each), swing-foot forces = 0, friction, torque, joint limits.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
import scipy.linalg as sla

from idqp_core import Config, IDQPController, smooth

G = 9.81
PHASE_IDS = {"stand": 0, "ds": 1, "ss": 2, "stand_end": 3}


@dataclass
class WalkConfig(Config):
    n_steps: int = 8
    step_len: float = 0.10        # x distance between the feet after a step
    t_ss: float = 0.5             # single support duration
    t_ds: float = 0.2             # double support duration
    step_h: float = 0.05          # swing foot apex height
    t_init: float = 1.0           # standing before the first step
    t_end_ds: float = 0.6         # final double support (ZMP back to the middle)
    t_end_stand: float = 1.5      # standing after the last step
    sole_off: tuple = (0.035, 0.0)   # sole center relative to the ankle frame origin (x, y)
    # ZMP preview control
    preview_dt: float = 0.005
    preview_T: float = 1.6
    preview_R: float = 1e-6
    # swing foot and pelvis tasks
    kp_sw: float = 400.0; kd_sw: float = 40.0
    w_sw: float = 100.0; w_pelv: float = 10.0
    # optional lateral push during the walk (time = after `push_after_steps` steps + 0.3 s)
    push: bool = False
    push_f: float = 40.0
    push_after_steps: int = 3
    push_dur: float = 0.15
    pushes: list = field(default_factory=list)   # filled in by the planner


class WalkController(IDQPController):
    def __init__(self, cfg: WalkConfig | None = None):
        cfg = cfg or WalkConfig()
        super().__init__(cfg)                  # model, nominal pose, constant constraints, reset()
        self.FEET = self.FOOT_BODIES           # 0 = left, 1 = right
        self.corner_foot = np.array([self.FEET.index(b) for b, _ in self.corners])
        self.SOLE_OFF = np.array(cfg.sole_off)
        self.plan()

    # ================================================================== planning (done once)
    def plan(self):
        cfg, d = self.cfg, self.d
        mujoco.mj_forward(self.m, d)           # state is q_nom after reset()
        foot0 = [d.xpos[b].copy() for b in self.FEET]
        self.foot0 = foot0
        c0 = self.c0
        self.ZC = c0[2]
        self.steps = self._plan_footsteps()
        self._plan_phases()
        self._plan_preview()
        self._plan_pushes()

    def _plan_footsteps(self):
        """1. footsteps: list of (swing foot, from, to) for the ankle frame origin."""
        cfg, foot0 = self.cfg, self.foot0
        steps, pos, swing = [], [foot0[0].copy(), foot0[1].copy()], 0       # start with the left foot
        for k in range(cfg.n_steps):
            tgt = pos[swing].copy()
            tgt[0] = pos[1 - swing][0] + (0.0 if k == cfg.n_steps - 1 else cfg.step_len)   # last step: feet together
            steps.append((swing, pos[swing].copy(), tgt.copy()))
            pos[swing] = tgt
            swing = 1 - swing
        return steps

    def _plan_phases(self):
        """2. phases: (t0, t1, kind, info); kind in stand / ds / ss / stand_end. Also the ZMP anchor points."""
        cfg, foot0, c0, steps = self.cfg, self.foot0, self.c0, self.steps
        phases = [(0.0, cfg.t_init, "stand", None)]
        t = cfg.t_init
        zmp_pts = [c0[:2].copy()]
        cur = [foot0[0][:2].copy(), foot0[1][:2].copy()]
        for sw, p_from, p_to in steps:
            z_to = cur[1 - sw] + self.SOLE_OFF                     # ZMP moves onto the stance foot
            phases.append((t, t + cfg.t_ds, "ds", (zmp_pts[-1].copy(), z_to.copy()))); t += cfg.t_ds
            phases.append((t, t + cfg.t_ss, "ss", (sw, p_from, p_to, z_to.copy()))); t += cfg.t_ss
            zmp_pts.append(z_to); cur[sw] = p_to[:2].copy()
        z_mid = (cur[0] + cur[1]) / 2 + self.SOLE_OFF
        z_mid[0] = (cur[0][0] + cur[1][0]) / 2 + (c0[0] - foot0[0][0])   # keep the initial CoM offset
        phases.append((t, t + cfg.t_end_ds, "ds", (zmp_pts[-1].copy(), z_mid.copy()))); t += cfg.t_end_ds
        phases.append((t, t + cfg.t_end_stand, "stand_end", z_mid.copy())); t += cfg.t_end_stand
        self.phases, self.T_TOTAL = phases, t
        self.footsteps = [p_to for _, _, p_to in steps]

    def _plan_preview(self):
        """3. ZMP preview control on the cart-table model, per axis -> CoM plan (x, y)."""
        cfg, c0 = self.cfg, self.c0
        DTP = cfg.preview_dt
        NPREV = int(cfg.preview_T / DTP)
        A = np.array([[1, DTP, DTP ** 2 / 2], [0, 1, DTP], [0, 0, 1]])
        B = np.array([[DTP ** 3 / 6], [DTP ** 2 / 2], [DTP]])
        C = np.array([[1, 0, -self.ZC / G]])
        At = np.block([[np.eye(1), C @ A], [np.zeros((3, 1)), A]])
        Bt = np.vstack([C @ B, B])
        Qt = np.diag([1.0, 0, 0, 0]); R = np.array([[cfg.preview_R]])
        P = sla.solve_discrete_are(At, Bt, Qt, R)
        Kt = np.linalg.solve(R + Bt.T @ P @ Bt, Bt.T @ P @ At)
        Gi, Gx = Kt[0, 0], Kt[0, 1:]
        Ac = At - Bt @ Kt
        X = -Ac.T @ P @ np.array([[1.0], [0], [0], [0]])
        Gd = np.zeros(NPREV); Gd[0] = -Gi
        for j in range(1, NPREV):
            Gd[j] = np.linalg.solve(R + Bt.T @ P @ Bt, Bt.T @ X)[0, 0]
            X = Ac.T @ X

        tp = np.arange(0, self.T_TOTAL + NPREV * DTP + DTP, DTP)
        pref = np.array([self.zmp_ref(min(ti, self.T_TOTAL - 1e-6)) for ti in tp])
        com_plan = np.zeros((len(tp), 3, 2))          # [k, (pos, vel, acc), axis]
        zmp_plan = np.zeros((len(tp), 2))
        for ax in range(2):
            x = np.array([c0[ax], 0, 0]); esum = 0.0
            for k in range(len(tp) - NPREV):
                pz = (C @ x)[0]; esum += pz - pref[k, ax]
                u = -Gi * esum - Gx @ x - Gd[1:] @ pref[k + 1:k + NPREV, ax]
                com_plan[k, :, ax] = x; zmp_plan[k, ax] = pz
                x = A @ x + B[:, 0] * u
        self.NPLAN = len(tp) - NPREV
        self.DTP = DTP
        self.tp, self.com_plan, self.zmp_plan = tp[:self.NPLAN], com_plan[:self.NPLAN], zmp_plan[:self.NPLAN]

    def _plan_pushes(self):
        """Optional push in the middle of the walk."""
        cfg = self.cfg
        if cfg.push:
            t_push = cfg.t_init + cfg.push_after_steps * (cfg.t_ds + cfg.t_ss) + 0.3
            cfg.pushes = [(t_push, cfg.push_dur, np.array([0, cfg.push_f, 0]))]
        else:
            cfg.pushes = []

    # ================================================================== references
    def phase_at(self, t):
        for ph in self.phases:
            if ph[0] <= t < ph[1]:
                return ph
        return self.phases[-1]

    def zmp_ref(self, t):
        t0, t1, kind, info = self.phase_at(t)
        if kind == "stand":
            return self.c0[:2].copy()
        if kind == "stand_end":
            return info.copy()
        if kind == "ds":
            a, b = info
            return a + (b - a) * smooth((t - t0) / (t1 - t0))
        return info[3].copy()

    def com_ref(self, t, h=None):
        k = min(int(t / self.DTP), self.NPLAN - 2); a = t / self.DTP - k
        a = min(a, 1.0)
        v = (1 - a) * self.com_plan[k] + a * self.com_plan[k + 1]
        p = np.array([v[0, 0], v[0, 1], self.ZC])
        return p, np.array([v[1, 0], v[1, 1], 0]), np.array([v[2, 0], v[2, 1], 0])

    def swing_pos(self, p_from, p_to, s):
        s = np.clip(s, 0, 1)
        p = (p_from + (p_to - p_from) * smooth(s)).copy()
        p[2] = p_from[2] + self.cfg.step_h * 16 * s ** 2 * (1 - s) ** 2   # lift off fast, no dragging
        return p

    def swing_ref(self, info, t0, t1, t, h=1e-3):
        _, a, b = info[:3]
        f = lambda tt: self.swing_pos(a, b, (tt - t0) / (t1 - t0))
        p, pp, pm = f(t), f(t + h), f(t - h)
        return p, (pp - pm) / (2 * h), (pp - 2 * p + pm) / (h * h)

    def contact_plan(self, t):
        t0, t1, kind, info = self.phase_at(t)
        contact = [True, True]
        if kind == "ss":
            contact[info[0]] = False
        return contact

    # ================================================================== jacobians
    def task_jacobians(self, data):
        """[J_left (6xnv), J_right (6xnv), Jcom, J_torso_rot, J_pelvis_rot]."""
        m, jp, jr = self.m, self._jp, self._jr
        out = []
        for b in self.FEET:
            mujoco.mj_jacBody(m, data, jp, jr, b); out.append(np.vstack([jp, jr]))
        Jcom = np.zeros((3, self.nv)); mujoco.mj_jacSubtreeCom(m, data, Jcom, self.PELVIS)
        out.append(Jcom)
        mujoco.mj_jacBody(m, data, jp, jr, self.TORSO); out.append(jr.copy())
        mujoco.mj_jacBody(m, data, jp, jr, self.PELVIS); out.append(jr.copy())
        return out

    def orient_err(self, body, target=np.array([1.0, 0, 0, 0])):
        e = np.zeros(3); mujoco.mju_subQuat(e, target, self.d.xquat[body])
        return self.d.xmat[body].reshape(3, 3) @ e

    # ================================================================== the QP
    def solve(self, t):
        m, d, cfg = self.m, self.d, self.cfg
        nv, nu, NX, NC, nf, iq = self.nv, self.nu, self.NX, self.NC, self.nf, self.iq
        mujoco.mj_forward(m, d)
        q, qd = d.qpos.copy(), d.qvel.copy()
        M = np.zeros((nv, nv)); mujoco.mj_fullM(m, d, M)
        h = d.qfrc_bias.copy()
        JL, JR, Jcom, Jt, Jpv = self.task_jacobians(d)
        dJL, dJR, dJcom, dJt, dJpv = self.dJqd(q, qd)
        Jfeet, dJfeet = [JL, JR], [dJL, dJR]
        Jc = self.corner_jac(d)

        t0, t1, kind, info = self.phase_at(t)
        contact = self.contact_plan(t)

        tasks = {}
        c, cd = d.subtree_com[self.PELVIS].copy(), Jcom @ qd
        cr, cdr, cddr = self.com_ref(t)
        a_com = cddr + cfg.kp_com * (cr - c) + cfg.kd_com * (cdr - cd)
        A = np.zeros((3, NX)); A[:, iq] = Jcom; tasks["com"] = (cfg.w_com, A, a_com - dJcom)
        err_t = self.orient_err(self.TORSO)
        a_rot = cfg.kp_rot * err_t - cfg.kd_rot * (Jt @ qd)
        A = np.zeros((3, NX)); A[:, iq] = Jt; tasks["rot"] = (cfg.w_rot, A, a_rot - dJt)
        a_pv = cfg.kp_rot * self.orient_err(self.PELVIS) - cfg.kd_rot * (Jpv @ qd)
        A = np.zeros((3, NX)); A[:, iq] = Jpv; tasks["pelvis"] = (cfg.w_pelv, A, a_pv - dJpv)

        sw_ref = sw_pos = None
        swing_foot = -1
        if kind == "ss":
            swing_foot = sw = info[0]
            J6, dJ6 = Jfeet[sw], dJfeet[sw]
            pr, vr, ar = self.swing_ref(info, t0, t1, t)
            sw_ref, sw_pos = pr, d.xpos[self.FEET[sw]].copy()
            a_lin = ar + cfg.kp_sw * (pr - sw_pos) + cfg.kd_sw * (vr - J6[:3] @ qd)
            a_ang = cfg.kp_sw * self.orient_err(self.FEET[sw]) - cfg.kd_sw * (J6[3:] @ qd)
            A = np.zeros((6, NX)); A[:, iq] = J6
            tasks["swing"] = (cfg.w_sw, A, np.concatenate([a_lin, a_ang]) - dJ6)

        a_post = cfg.kp_post * (self.q_nom[7:] - q[7:]) - cfg.kd_post * qd[6:]
        wj = np.sqrt(np.where(np.arange(nu) < 12, cfg.w_post_leg, cfg.w_post))
        A = np.zeros((nv - 6, NX)); A[:, 6:nv] = np.diag(wj); tasks["post"] = (1.0, A, wj * a_post)

        P = np.zeros((NX, NX)); g = np.zeros(NX)
        for w, A, b in tasks.values():
            P += 2 * w * A.T @ A; g -= 2 * w * A.T @ b
        P[iq, iq] += 2 * cfg.w_qdd * np.eye(nv)
        P[self.it, self.it] += 2 * cfg.w_tau * np.eye(nu)
        P[self.iff, self.iff] += 2 * cfg.w_f * np.eye(nf)

        # constraints; rows and names depend on which feet are in contact
        rows, lo, hi, names, groups = [], [], [], [], {}

        def add(gname, A_, l_, u_, nm):
            k = sum(r.shape[0] for r in rows)
            rows.append(A_); lo.append(l_); hi.append(u_); names.extend(nm)
            groups[gname] = slice(groups[gname].start, k + A_.shape[0]) if gname in groups else slice(k, k + A_.shape[0])

        add("dyn", np.hstack([M, -self.S.T, -Jc.T]), -h, -h, self.row_names[self.groups["dyn"]])
        feet_names = self.row_names[self.groups["feet"]]
        groups["feet"] = slice(sum(r.shape[0] for r in rows), sum(r.shape[0] for r in rows))
        for f in range(2):
            if contact[f]:
                Af = np.hstack([Jfeet[f], np.zeros((6, nu + nf))])
                bf = -dJfeet[f] - cfg.kf * (Jfeet[f] @ qd)
                add("feet", Af, bf, bf, feet_names[6 * f:6 * f + 6])
        u_fric = self.u_fric.copy()
        u_fric[4::5] = np.where(np.array([contact[self.corner_foot[i]] for i in range(NC)]), cfg.fz_max, 0.0)
        qj, vj = q[7:], qd[6:]
        qlo = 2 * (self.jlo - qj - vj * cfg.h_lim) / cfg.h_lim ** 2
        qhi = 2 * (self.jhi - qj - vj * cfg.h_lim) / cfg.h_lim ** 2
        qlo = np.minimum(qlo, qhi)
        add("tau", self.A_tau, -self.tau_max, self.tau_max, self.row_names[self.groups["tau"]])
        add("fric", self.A_fric, self.l_fric, u_fric, self.row_names[self.groups["fric"]])
        add("lim", self.A_lim, qlo, np.maximum(qhi, qlo), self.row_names[self.groups["lim"]])
        Acons, lo, hi = np.vstack(rows), np.concatenate(lo), np.concatenate(hi)

        base = dict(t=t, c=c, cd=cd, c_ref=cr, cdd_des=a_com, a_rot=a_rot, rot_err=err_t,
                    Jcom=Jcom, dJcom=dJcom, Jc=Jc,
                    phase=kind, phase_id=PHASE_IDS[kind], phase_s=(t - t0) / (t1 - t0),
                    contact_plan=np.array(contact), swing_foot=swing_foot,
                    swing_ref=sw_ref if sw_ref is not None else np.full(3, np.nan),
                    swing_pos=sw_pos if sw_pos is not None else np.full(3, np.nan),
                    zmp_ref=self.zmp_ref(t))
        return self.solve_qp(P, g, Acons, lo, hi, tasks, base, groups=groups, row_names=list(names))

    def log_extra(self):
        """Plan arrays stored next to the log (used by plot_g1_walk.py)."""
        return dict(tp=self.tp, com_plan=self.com_plan, zmp_plan=self.zmp_plan, T_TOTAL=self.T_TOTAL,
                    phases=np.array([[ph[0], ph[1], PHASE_IDS[ph[2]]] for ph in self.phases]),
                    footsteps=np.array([[sw, *p_to] for sw, _, p_to in self.steps]),
                    sole_off=self.SOLE_OFF, foot0=np.array(self.foot0))
