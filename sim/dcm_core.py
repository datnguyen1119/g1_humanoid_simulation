"""G1 walking with DCM (divergent component of motion / capture point) on a flat floor.

Pipeline (each item says when it runs):
  1. footsteps + phase timing          same as walk_core.WalkController (plan, and again at every touchdown)
  2. ZMP reference r_ref(t)             same as walk_core: constant on the stance sole in single support,
                                         quintic blend in double support
  3. DCM reference xi_ref(t)            closed form, integrated BACKWARD from the end (xi_end = final ZMP):
                                           xi' = w (xi - r)   ->  xi(t) = P(t) + e^{w (t - t1)} (xi(t1) - P(t1))
                                         with P(t) = sum_j r^(j)(t) / w^j (exact for the polynomial ZMP)
  4. DCM feedback (every 2 ms)          xi = c + c'/w (measured);  e = xi - xi_ref
                                           r_cmd = r_ref + (1 + k/w) e      ->  e' = -k e
                                         r_cmd is clipped to the support polygon shrunk by cop_inset
  5. step adjustment (single support)   per axis, solve a tiny QP over (CoP r, landing point p):
                                           p + b_nom = r + E (xi - r),   E = e^{w T_remaining}
                                           min w_cop (r - r_nom)^2 + w_step (p - p_nom)^2
                                           r in the stance sole box, p in a reachable box
                                         -> CoP absorbs small errors, the step absorbs the rest.
                                         Frozen in the last (1 - s_freeze) of the swing; committed at
                                         touchdown, after which steps 1-3 are re-planned.
  6. ID-QP (same constraints as walk_core); only the CoM task changes:
                                           a_com,xy = w^2 (c - r_cmd)     (LIPM: the CoP of the QP forces ~ r_cmd)
                                           a_com,z  = Kp (z_c - c_z) - Kd c'_z

walk_core.py is not modified: the QP is copied here (solve) with the CoM task and the swing reference
replaced. With step_adjust=False and k_dcm feedback, everything else is the same walking controller.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.signal import lfilter

from idqp_monitor import _add_geom, _add_line, convex_hull
from walk_core import G, PHASE_IDS, WalkConfig, WalkController

SOLE_X = (-0.05, 0.12)     # contact corners in x, relative to the ankle frame origin (m)
SOLE_Y = 0.025             # half width at the heel (toe is 0.03: use the smaller one)
DT_COM = 0.002             # grid of the CoM reference (s)
MODES = {0: "feedback", 1: "step adjust", 2: "frozen"}


@dataclass
class DCMConfig(WalkConfig):
    k_dcm: float = 6.0                 # DCM error decay rate (1/s): e' = -k e
    cop_inset: float = 0.0             # r_cmd margin inside the support polygon (m); < 0: CMP may leave it (L-dot)
    step_adjust: bool = True
    w_cop: float = 1.0                 # cost of moving the CoP away from its plan
    w_step: float = 1.0                # cost of moving the landing point away from its plan
    s_freeze: float = 0.8              # no more landing changes after this fraction of the swing
    shift_following: bool = True       # after an adjusted step, move all later steps by the same shift
    step_dx: tuple = (-0.10, 0.30)     # landing x relative to the stance ankle (m)
    step_width: tuple = (0.17, 0.38)   # |y| distance between the feet (m): no crossing, no splits
    push_t: float | None = None        # push start time; None = walk_core rule (after push_after_steps + 0.3 s)
    dcm_err_warn: float = 0.02         # DCM_ERR_HIGH event threshold (m)


def _sigma_derivs(s):
    """smooth(s) = 10s^3 - 15s^4 + 6s^5 and its derivatives d^j/ds^j, j = 0..5 (rows)."""
    s = np.asarray(s, float)
    return np.array([s ** 3 * (10 - 15 * s + 6 * s * s),
                     30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4,
                     60 * s - 180 * s ** 2 + 120 * s ** 3,
                     60 - 360 * s + 360 * s ** 2,
                     -360 + 720 * s,
                     np.full_like(s, 720.0)])


def quintic(p0, v0, a0, p1, T):
    """Coefficients (6 x dim) of p(tau): (p0, v0, a0) at tau = 0, (p1, 0, 0) at tau = T."""
    p0, v0, a0, p1 = map(np.asarray, (p0, v0, a0, p1))
    c0, c1, c2 = p0, v0, a0 / 2
    rhs = np.vstack([p1 - (c0 + c1 * T + c2 * T ** 2), -(c1 + 2 * c2 * T), -2 * c2])
    M = np.array([[T ** 3, T ** 4, T ** 5], [3 * T ** 2, 4 * T ** 3, 5 * T ** 4], [6 * T, 12 * T ** 2, 20 * T ** 3]])
    return np.vstack([c0, c1, c2, np.linalg.solve(M, rhs)])


def quintic_eval(C, tau):
    p = (tau ** np.arange(6)) @ C
    v = (np.arange(1, 6) * tau ** np.arange(5)) @ C[1:]
    a = (np.array([2, 6, 12, 20]) * tau ** np.arange(4)) @ C[2:]
    return p, v, a


def shrink_polygon(hull, d):
    """Convex CCW polygon with every edge moved inward by d: list of (normal, offset) and vertices."""
    n = len(hull)
    lines = []
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        e = b - a
        nrm = np.array([e[1], -e[0]]) / np.linalg.norm(e)         # outward for CCW
        lines.append((nrm, nrm @ a - d))
    verts = []
    for i in range(n):
        (n1, c1), (n2, c2) = lines[i - 1], lines[i]
        A = np.array([n1, n2])
        if abs(np.linalg.det(A)) > 1e-9:
            verts.append(np.linalg.solve(A, [c1, c2]))
    return lines, np.array(verts)


def project_polygon(p, lines, verts):
    """p itself if inside the half-planes, otherwise the closest point of the polygon boundary."""
    if all(nrm @ p <= c + 1e-12 for nrm, c in lines):
        return p.copy()
    best, bd = p.copy(), np.inf
    for a, b in zip(verts, np.roll(verts, -1, axis=0)):
        e = b - a
        s = np.clip((p - a) @ e / max(e @ e, 1e-12), 0, 1)
        q = a + s * e
        if np.linalg.norm(p - q) < bd:
            best, bd = q, np.linalg.norm(p - q)
    return best


class DCMWalkController(WalkController):
    def __init__(self, cfg: DCMConfig | None = None):
        cfg = cfg or DCMConfig()
        super().__init__(cfg)                       # -> plan(): footsteps, phases, _plan_preview (= DCM), pushes
        self.steps_nominal = copy.deepcopy(self.steps)
        self._reset_runtime()

    # ================================================================== planning
    def _plan_preview(self):                        # called by WalkController.plan(): DCM plan instead
        self.omega = np.sqrt(G / self.ZC)
        self._plan_dcm(0.0, self.c0[:2].copy())

    def _plan_pushes(self):
        cfg = self.cfg
        if cfg.push:
            t_push = cfg.push_t if cfg.push_t is not None else \
                cfg.t_init + cfg.push_after_steps * (cfg.t_ds + cfg.t_ss) + 0.3
            cfg.pushes = [(t_push, cfg.push_dur, np.array([0, cfg.push_f, 0]))]
        else:
            cfg.pushes = []

    def _zmp_terms(self, ph, t):
        """ZMP r(t) and P(t) = sum_j r^(j)(t) / w^j inside phase ph, for an array of times."""
        t = np.atleast_1d(np.asarray(t, float))
        t0, t1, kind, info = ph
        if kind == "ds":
            a, b = info
            T = t1 - t0
            S = _sigma_derivs(np.clip((t - t0) / T, 0, 1))
            r = a + np.outer(S[0], b - a)
            coef = sum(S[j] / (T * self.omega) ** j for j in range(1, 6))
            return r, r + np.outer(coef, b - a)
        r0 = self.c0[:2] if kind == "stand" else (info if kind == "stand_end" else info[3])
        r = np.tile(r0, (len(t), 1))
        return r, r.copy()

    def _xi_phase(self, i, t):
        ph = self.phases[i]
        _, P = self._zmp_terms(ph, t)
        _, P1 = self._zmp_terms(ph, ph[1])
        tt = np.atleast_1d(np.asarray(t, float))
        return P + np.exp(self.omega * (tt - ph[1]))[:, None] * (self.xi_end[i] - P1[0])

    def _plan_dcm(self, t_from, c_from):
        """DCM at the end of every phase (backward pass), then the CoM reference from t_from on."""
        n = len(self.phases)
        self.xi_end = [None] * n
        xi = self._zmp_terms(self.phases[-1], self.phases[-1][1])[0][0]   # stand still on the final ZMP
        for i in range(n - 1, -1, -1):
            self.xi_end[i] = xi
            xi = self._xi_phase(i, self.phases[i][0])[0]
        self.xi_start = xi
        # CoM reference: c' = w (xi_ref - c), exact per step for xi held at the midpoint
        tg = np.arange(t_from, self.T_TOTAL + DT_COM, DT_COM)
        xi_m = self._xi_on(tg[:-1] + DT_COM / 2)
        alpha = np.exp(-self.omega * DT_COM)
        cg = np.empty((len(tg), 2)); cg[0] = c_from
        for ax in range(2):
            cg[1:, ax] = lfilter([1 - alpha], [1, -alpha], xi_m[:, ax], zi=[alpha * c_from[ax]])[0]
        self.tc, self.cg = tg, cg

    def _xi_on(self, t):
        out = np.empty((len(t), 2))
        for i, ph in enumerate(self.phases):
            mk = (t >= ph[0]) & ((t < ph[1]) | (i == len(self.phases) - 1))
            if mk.any():
                out[mk] = self._xi_phase(i, t[mk])
        return out

    def _phase_index(self, t):
        for i, ph in enumerate(self.phases):
            if ph[0] <= t < ph[1]:
                return i
        return len(self.phases) - 1

    def dcm_ref(self, t):
        """(xi_ref, xi_ref', r_ref) at time t."""
        i = self._phase_index(t)
        xi = self._xi_phase(i, t)[0]
        r = self._zmp_terms(self.phases[i], t)[0][0]
        return xi, self.omega * (xi - r), r

    def com_ref(self, t, h=None):
        c = np.array([np.interp(t, self.tc, self.cg[:, ax]) for ax in range(2)])
        xi, _, r = self.dcm_ref(t)
        w = self.omega
        return (np.array([c[0], c[1], self.ZC]), np.r_[w * (xi - c), 0.0], np.r_[w * w * (c - r), 0.0])

    def ss_index(self, t):
        """Index of the step whose swing contains t (None outside single support)."""
        k = -1
        for ph in self.phases:
            if ph[2] == "ss":
                k += 1
                if ph[0] <= t < ph[1]:
                    return k
        return None

    # ================================================================== runtime state
    def reset(self):
        super().reset()
        if hasattr(self, "steps_nominal"):          # (IDQPController.__init__ calls reset before planning)
            self.steps = copy.deepcopy(self.steps_nominal)
            self._plan_phases()
            self._plan_pushes()
            self._plan_dcm(0.0, self.c0[:2].copy())
            self._reset_runtime()

    def _reset_runtime(self):
        self.k_ss = -1             # step index of the swing in progress (step adjustment)
        self.p_cmd = None          # its commanded landing point (ankle xy)
        self.sw_quint = None       # (t_start, coefficients, duration) of the swing xy quintic
        self.commits = []          # (t, k, dx, dy): landing shift vs the nominal plan, at touchdown
        nan2 = np.full(2, np.nan)
        self.dcm_state = dict(dcm=nan2, dcm_ref=nan2, dcm_err=np.nan, V=np.nan, r_ref=nan2, r_cmd_raw=nan2,
                              r_cmd=nan2, cop_sat=0.0, step_shift=np.zeros(2), dcm_mode=0, p_cmd=nan2)

    def _set_landing(self, k, xy):
        """Move the landing point of step k; the next swing of the same foot starts from there."""
        self.steps[k][2][:2] = xy
        if k + 2 < len(self.steps):
            self.steps[k + 2][1][:2] = xy

    def _commit(self, t, c_ref_now):
        """Touchdown of step k_ss: keep its landing point, re-fit the later steps, re-plan ZMP / DCM / CoM."""
        cfg, k = self.cfg, self.k_ss
        self._set_landing(k, self.p_cmd)
        prev = self.p_cmd.copy()
        # later steps: nominal pattern moved by this step's shift (walk on along the new line), or kept in place
        offset = self.p_cmd - self.steps_nominal[k][2][:2] if cfg.shift_following else np.zeros(2)
        for j in range(k + 1, len(self.steps)):
            sw = self.steps[j][0]
            p = self.steps_nominal[j][2][:2] + offset
            p[0] = np.clip(p[0], prev[0] + cfg.step_dx[0], prev[0] + cfg.step_dx[1])
            wl, wh = cfg.step_width
            p[1] = np.clip(p[1], prev[1] + wl, prev[1] + wh) if sw == 0 else np.clip(p[1], prev[1] - wh, prev[1] - wl)
            self._set_landing(j, p)
            prev = p
        shift = self.p_cmd - self.steps_nominal[k][2][:2]
        self.commits.append((t, k, *shift))
        self._plan_phases()
        self._plan_dcm(t, c_ref_now)
        self.k_ss, self.p_cmd, self.sw_quint = -1, None, None

    def _step_opt(self, t, t1, sw, xi, xi_r, r_ref, a_s, p_nom):
        """CoP r and landing point p (ankle) for the rest of this swing; see the module docstring."""
        cfg, off = self.cfg, self.SOLE_OFF
        E = np.exp(self.omega * (t1 - t))
        b = r_ref + E * (xi_r - r_ref) - (p_nom + off)      # nominal DCM - sole centre at touchdown
        ins = cfg.cop_inset
        r_lo = a_s + [SOLE_X[0] + ins, -SOLE_Y + ins]
        r_hi = a_s + [SOLE_X[1] - ins, SOLE_Y - ins]
        wl, wh = cfg.step_width
        ylo, yhi = (a_s[1] + wl, a_s[1] + wh) if sw == 0 else (a_s[1] - wh, a_s[1] - wl)
        p_lo = np.array([a_s[0] + cfg.step_dx[0], ylo])
        p_hi = np.array([a_s[0] + cfg.step_dx[1], yhi])
        r, p = np.empty(2), np.empty(2)
        A = 1 - E                                            # p(r) = A r + B, A < 0
        for ax in range(2):
            B = E * xi[ax] - b[ax] - off[ax]
            d0 = A * r_ref[ax] + B - p_nom[ax]
            r_star = r_ref[ax] - cfg.w_step * A * d0 / (cfg.w_cop + cfg.w_step * A * A)
            lo_r, hi_r = (p_hi[ax] - B) / A, (p_lo[ax] - B) / A      # r interval that keeps p reachable
            if lo_r > r_hi[ax]:
                rr = r_hi[ax]                                # not capturable in one step: saturate both
            elif hi_r < r_lo[ax]:
                rr = r_lo[ax]
            else:
                rr = np.clip(r_star, max(lo_r, r_lo[ax]), min(hi_r, r_hi[ax]))
            r[ax] = rr
            p[ax] = np.clip(A * rr + B, p_lo[ax], p_hi[ax])
        return r, p

    def _support(self, contact):
        pts = self.corner_points(self.d)
        sel = [pts[i, :2] for i in range(self.NC) if contact[self.corner_foot[i]]]
        return shrink_polygon(convex_hull(sel), self.cfg.cop_inset)

    def swing_ref(self, info, t0, t1, t, h=1e-3):
        pr, vr, ar = super().swing_ref(info, t0, t1, t, h)      # z profile (and xy without step adjustment)
        if self.cfg.step_adjust and self.sw_quint is not None:
            tq, C, T = self.sw_quint
            pr[:2], vr[:2], ar[:2] = quintic_eval(C, float(np.clip(t - tq, 0, T)))
        return pr, vr, ar

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
        c, cd = d.subtree_com[self.PELVIS].copy(), Jcom @ qd

        # ---- touchdown of an adjusted step: commit it and re-plan before reading the references
        k_now = self.ss_index(t)
        if self.k_ss >= 0 and k_now != self.k_ss:
            self._commit(t, self.com_ref(t)[0][:2])
        t0, t1, kind, info = self.phase_at(t)
        contact = self.contact_plan(t)

        # ---- DCM feedback -> CoP command
        w = self.omega
        xi = c[:2] + cd[:2] / w
        xi_r, _, r_ref = self.dcm_ref(t)
        e = xi - xi_r
        r_raw = r_ref + (1 + cfg.k_dcm / w) * e
        mode, shift = 0, np.zeros(2)
        if kind == "ss" and cfg.step_adjust:
            sw = info[0]
            if self.k_ss != k_now:                           # a new swing starts
                self.k_ss, self.p_cmd = k_now, info[2][:2].copy()
                self.sw_quint = (t0, quintic(info[1][:2], np.zeros(2), np.zeros(2), self.p_cmd, t1 - t0), t1 - t0)
            if (t - t0) / (t1 - t0) < cfg.s_freeze:
                mode = 1
                a_s = d.xpos[self.FEET[1 - sw]][:2]
                p_nom = self.steps[k_now][2][:2]
                r_raw, p_new = self._step_opt(t, t1, sw, xi, xi_r, r_ref, a_s, p_nom)
                if np.linalg.norm(p_new - self.p_cmd) > 2e-4:   # re-plan the swing xy from its current reference
                    tq, C, T = self.sw_quint
                    p0, v0, a0 = quintic_eval(C, float(np.clip(t - tq, 0, T)))
                    self.sw_quint = (t, quintic(p0, v0, a0, p_new, t1 - t), t1 - t)
                    self.p_cmd = p_new
            else:
                mode = 2
            shift = self.p_cmd - self.steps_nominal[k_now][2][:2]
        lines, verts = self._support(contact)
        r_cmd = project_polygon(r_raw, lines, verts)
        self.dcm_state = dict(dcm=xi, dcm_ref=xi_r, dcm_err=float(np.linalg.norm(e)), V=float(0.5 * e @ e),
                              r_ref=r_ref, r_cmd_raw=r_raw, r_cmd=r_cmd,
                              cop_sat=float(np.linalg.norm(r_cmd - r_raw) > 1e-4), step_shift=shift,
                              dcm_mode=mode, p_cmd=self.p_cmd.copy() if self.p_cmd is not None else np.full(2, np.nan))

        tasks = {}
        cr = self.com_ref(t)[0]
        a_com = np.r_[w * w * (c[:2] - r_cmd), cfg.kp_com * (self.ZC - c[2]) - cfg.kd_com * cd[2]]
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
        for wt, A, b in tasks.values():
            P += 2 * wt * A.T @ A; g -= 2 * wt * A.T @ b
        P[iq, iq] += 2 * cfg.w_qdd * np.eye(nv)
        P[self.it, self.it] += 2 * cfg.w_tau * np.eye(nu)
        P[self.iff, self.iff] += 2 * cfg.w_f * np.eye(nf)

        # constraints (identical to walk_core): rows and names depend on which feet are in contact
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
                    zmp_ref=r_ref)
        return self.solve_qp(P, g, Acons, lo, hi, tasks, base, groups=groups, row_names=list(names))

    # ================================================================== monitor hooks
    def touchdown_info(self, swing_foot, logger):
        if not self.cfg.step_adjust or not self.commits:
            return "no step adjustment"
        _, k, dx, dy = self.commits[-1]
        return f"landing shift vs plan ({dx * 1e3:+.0f}, {dy * 1e3:+.0f}) mm"

    def overlay_rows(self, st):
        e = st.get("dcm_err", np.nan)
        rows = [("DCM err", f"{e * 1e3:.1f} mm" if np.isfinite(e) else "-"),
                ("CoP cmd", "SATURATED" if st.get("cop_sat", 0) else "inside"),
                ("DCM mode", MODES[int(st.get("dcm_mode", 0))])]
        if self.cfg.step_adjust:
            s = st.get("step_shift", np.zeros(2))
            rows.append(("step shift", f"{s[0] * 1e3:+.0f}, {s[1] * 1e3:+.0f} mm"))
        return rows

    def extra_markers(self, scn, st):
        """CP (measured DCM) red, DCM ref pink, CoP command orange, line CoP(sim) -> CP, landing target."""
        z = 0.004
        xi = st.get("dcm")
        if xi is None or not np.all(np.isfinite(xi)):
            return
        _add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.02, 0, 0], [xi[0], xi[1], z], [1, 0.1, 0.1, 1])          # CP
        xr = st["dcm_ref"]
        _add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.011, 0, 0], [xr[0], xr[1], z], [1, 0.6, 0.7, 0.9])       # CP ref
        rc = st["r_cmd"]
        _add_geom(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, [0.012, 0.003, 0], [rc[0], rc[1], z],
                  [1, 0.5, 0, 1] if not st["cop_sat"] else [1, 0, 0, 1])                                        # CoP cmd
        if np.all(np.isfinite(st["cop_sim"])):
            p = st["cop_sim"]
            _add_line(scn, [p[0], p[1], z], [xi[0], xi[1], z], [1, 0.2, 0.2, 0.9], 0.003)                       # CoP -> CP
        pc = st.get("p_cmd")
        if pc is not None and np.all(np.isfinite(pc)):
            _add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [0.085, 0.03, 0.002],
                      [pc[0] + self.SOLE_OFF[0], pc[1] + self.SOLE_OFF[1], 0.002], [0.1, 1, 0.1, 0.45])          # landing target
        if self.cfg.step_adjust:
            for _, _, p_to in self.steps_nominal:
                _add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [0.085, 0.03, 0.0005],
                          [p_to[0] + self.SOLE_OFF[0], p_to[1] + self.SOLE_OFF[1], 0.0005], [0.6, 0.6, 0.6, 0.25])

    def log_extra(self):
        return dict(omega=self.omega, k_dcm=self.cfg.k_dcm, step_adjust=self.cfg.step_adjust, T_TOTAL=self.T_TOTAL,
                    phases=np.array([[ph[0], ph[1], PHASE_IDS[ph[2]]] for ph in self.phases]),
                    footsteps=np.array([[sw, *p_to] for sw, _, p_to in self.steps]),
                    footsteps_nominal=np.array([[sw, *p_to] for sw, _, p_to in self.steps_nominal]),
                    commits=np.array(self.commits).reshape(-1, 4), sole_off=self.SOLE_OFF, foot0=np.array(self.foot0),
                    cop_inset=self.cfg.cop_inset)
