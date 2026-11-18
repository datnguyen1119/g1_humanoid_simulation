"""G1 walking with a linear ZMP MPC (QP solved by OSQP) instead of ZMP preview control. Flat floor, fixed footsteps.

Pipeline (each item says when it runs):
  1. footsteps + phase timing + ZMP reference z_ref(t)    same as walk_core.WalkController (plan, once)
  2. support polygon of every phase                       from the PLANNED feet (sole box), shrunk by zmp_margin
  3. DCM reference xi_ref(t)                              backward from the end of the plan:
                                                            xi_k = a xi_{k+1} + (1 - a) z_mean,   a = e^{-w dt}
  4. MPC (every replan_dt)                                per axis, state x = [c, c', c''], input u = jerk,
                                                          piecewise constant over T (mpc_T), horizon N:
        c''(s) = c''_k + u_k s,  c'(s) = c'_k + c''_k s + u_k s^2/2,  c(s) = c_k + c'_k s + c''_k s^2/2 + u_k s^3/6
        ->  x_{k+1} = A x_k + B u_k,   z_k = c_k - (h/g) c''_k
        ->  Z = Pzs x0 + Pzu U  (condensed over the horizon, same for C and C')
        min_U   alpha ||U||^2 + gamma ||Z - Z_ref||^2                     (Wieber 2006; = preview cost of Kajita)
        s.t.    n_j . z_k <= d_j          every sample k, every edge j of the support polygon at t + kT
                |xi_N - xi_ref(t + NT)| <= term_eps                       (xi_N = c_N + c'_N / w)
     x0 = measured (c, c') + c'' of the previous plan (feedback=True), or the previous plan only (open loop).
     QP failure: solve again without the terminal constraint, then keep the previous plan.
  5. ID-QP: walk_core.WalkController.solve unchanged; com_ref(t) integrates the MPC jerk exactly
     (the CoM task is a = c''_ref + Kp (c_ref - c) + Kd (c'_ref - c')).

walk_core.py is not modified. The preview plan is still computed (log / comparison only).
"""
from __future__ import annotations

import bisect
import time
from dataclasses import dataclass

import mujoco
import numpy as np
import osqp
import scipy.sparse as sp

from dcm_core import SOLE_X, SOLE_Y, shrink_polygon
from idqp_core import OK, WARN
from idqp_monitor import _add_geom, _add_line, convex_hull
from walk_core import G, PHASE_IDS, WalkConfig, WalkController

STATUS = {0: "ok", 1: "no terminal", 2: "hold plan"}
MAX_EDGES = 8                  # rows reserved per horizon sample (SS box: 4, DS hull: <= 6)


@dataclass
class MPCConfig(WalkConfig):
    mpc_T: float = 0.05                # sampling time of the MPC (s); jerk is constant over T. 0.1 s is too coarse
                                       # for the 0.2 s double support: the ZMP crosses phase boundaries between samples
    mpc_N: int = 32                    # horizon length (samples): N T = 1.6 s, same as the preview
    alpha: float = 1e-6                # weight of the jerk
    gamma: float = 1.0                 # weight of the ZMP tracking error
    zmp_margin: float = 0.005          # ZMP constraint margin inside the support polygon (m)
    use_terminal: bool = True          # |xi_N - xi_ref| <= term_eps
    term_eps: float = 0.01             # (m)
    feedback: bool = True              # x0 from the measured CoM (closed loop); False: from the previous plan
    replan_dt: float = 0.02            # MPC period (s)
    push_t: float | None = None        # push start time; None = walk_core rule (after push_after_steps + 0.3 s)
    dcm_err_warn: float = 0.02         # DCM_ERR_HIGH event threshold (m), for comparison with dcm_core


def lipm_condensed(T, N, hg):
    """Matrices of the horizon k = 1..N: pos C = Ps x0 + Pu U, vel, acc, ZMP Z = Zs x0 + Zu U (one axis)."""
    A = np.array([[1, T, T * T / 2], [0, 1, T], [0, 0, 1]])
    B = np.array([T ** 3 / 6, T * T / 2, T])
    Xs = np.zeros((N, 3, 3)); Xu = np.zeros((N, 3, N))
    Ak = np.eye(3)
    for k in range(N):                                # row k is the state at sample k + 1
        Ak = A @ Ak
        Xs[k] = Ak
        for j in range(k + 1):
            Xu[k, :, j] = np.linalg.matrix_power(A, k - j) @ B
    M = dict(A=A, B=B, Ps=Xs[:, 0], Pu=Xu[:, 0], Vs=Xs[:, 1], Vu=Xu[:, 1], As=Xs[:, 2], Au=Xu[:, 2])
    M["Zs"] = M["Ps"] - hg * M["As"]
    M["Zu"] = M["Pu"] - hg * M["Au"]
    return M


def jerk_eval(x, u, s):
    """State after s seconds of constant jerk u from x = [c, c', c''] (any leading shape)."""
    c, v, a = x[..., 0], x[..., 1], x[..., 2]
    return np.stack([c + v * s + a * s * s / 2 + u * s ** 3 / 6, v + a * s + u * s * s / 2, a + u * s], axis=-1)


class MPCWalkController(WalkController):
    def __init__(self, cfg: MPCConfig | None = None):
        cfg = cfg or MPCConfig()
        super().__init__(cfg)                       # -> plan(): footsteps, phases, preview (nominal), pushes
        self._build_mpc()
        self._reset_runtime()

    # ================================================================== planning (once)
    def _plan_pushes(self):
        cfg = self.cfg
        if cfg.push:
            t_push = cfg.push_t if cfg.push_t is not None else \
                cfg.t_init + cfg.push_after_steps * (cfg.t_ds + cfg.t_ss) + 0.3
            cfg.pushes = [(t_push, cfg.push_dur, np.array([0, cfg.push_f, 0]))]
        else:
            cfg.pushes = []

    def _build_mpc(self):
        cfg = self.cfg
        N = cfg.mpc_N
        self.omega = np.sqrt(G / self.ZC)
        self.hg = self.ZC / G
        self.M = lipm_condensed(cfg.mpc_T, N, self.hg)
        Zu = self.M["Zu"]
        H1 = cfg.alpha * np.eye(N) + cfg.gamma * Zu.T @ Zu
        self.P_mpc = sp.csc_matrix(np.triu(2 * np.kron(np.eye(2), H1)))      # x and y share the same block
        # xi_N = c_N + c'_N / w
        self.Xi_s = self.M["Ps"][-1] + self.M["Vs"][-1] / self.omega
        self.Xi_u = self.M["Pu"][-1] + self.M["Vu"][-1] / self.omega
        self._plan_supports()
        self._plan_xi_ref()

    def _plan_supports(self):
        """Support polygon (half-planes n . p <= d) of each phase, from the planned feet."""
        cur = [self.foot0[0][:2].copy(), self.foot0[1][:2].copy()]
        self.sup_t0, self.sup = [], []
        for t0, t1, kind, info in self.phases:
            feet = [cur[1 - info[0]]] if kind == "ss" else [cur[0], cur[1]]
            pts = [f + np.array([x, y]) for f in feet for x in SOLE_X for y in (-SOLE_Y, SOLE_Y)]
            hull = convex_hull(pts)
            lines, verts = shrink_polygon(hull, self.cfg.zmp_margin)
            self.sup_t0.append(t0)
            self.sup.append((np.array([n for n, _ in lines]), np.array([d for _, d in lines]), verts, hull))
            if kind == "ss":
                cur[info[0]] = info[2][:2].copy()

    def support_at(self, t):
        i = min(max(bisect.bisect_right(self.sup_t0, t) - 1, 0), len(self.sup) - 1)
        return self.sup[i]

    def _plan_xi_ref(self, dt=0.005):
        cfg = self.cfg
        tg = np.arange(0.0, self.T_TOTAL + (cfg.mpc_N + 2) * cfg.mpc_T + dt, dt)
        z = np.array([self.zmp_ref(min(t, self.T_TOTAL - 1e-6)) for t in tg])
        a = np.exp(-self.omega * dt)
        xi = np.empty_like(z); xi[-1] = z[-1]
        for k in range(len(tg) - 2, -1, -1):
            xi[k] = a * xi[k + 1] + (1 - a) * 0.5 * (z[k] + z[k + 1])
        self.xi_tg, self.xi_grid = tg, xi

    def xi_ref(self, t):
        return np.array([np.interp(t, self.xi_tg, self.xi_grid[:, a]) for a in range(2)])

    def zmp_ref_c(self, t):
        return self.zmp_ref(min(t, self.T_TOTAL - 1e-6))

    # ================================================================== runtime
    def reset(self):
        super().reset()
        if hasattr(self, "M"):                      # not during IDQPController.__init__
            self._reset_runtime()

    def _reset_runtime(self):
        N = self.cfg.mpc_N
        self.plan_t = 0.0
        self.plan_x0 = np.array([[self.c0[0], 0, 0], [self.c0[1], 0, 0]], dtype=float)
        self.plan_U = np.zeros((2, N))
        self._plan_states()
        self.next_replan = 0.0
        self.n_status = {0: 0, 1: 0, 2: 0}
        self.mpc_ms = []
        nan2, nanN = np.full(2, np.nan), np.full((N, 2), np.nan)
        self.mpc_state = dict(mpc_status=0, mpc_ms=0.0, mpc_iter=0, zmp_pred=nanN, com_pred=nanN,
                              zmp_active=0, term_active=0, xi_N=nan2, xi_ref_N=nan2,
                              xi=nan2, xi_ref=nan2, dcm_err=0.0, zmp_mpc=nan2)

    def _plan_states(self):
        """States of the current plan at its sample times (N + 1, axis, 3)."""
        T = self.cfg.mpc_T
        X = [self.plan_x0]
        for k in range(self.cfg.mpc_N):
            X.append(jerk_eval(X[-1], self.plan_U[:, k], T))
        self.plan_X = np.array(X)

    def plan_eval(self, t):
        """(axis, [c, c', c'']) of the current MPC plan at time t (exact jerk integration)."""
        T, N = self.cfg.mpc_T, self.cfg.mpc_N
        tau = max(t - self.plan_t, 0.0)
        k = min(int(tau / T + 1e-9), N - 1)
        return jerk_eval(self.plan_X[k], self.plan_U[:, k], tau - k * T)

    def com_ref(self, t, h=None):
        v = self.plan_eval(t)
        return (np.array([v[0, 0], v[1, 0], self.ZC]), np.array([v[0, 1], v[1, 1], 0.0]),
                np.array([v[0, 2], v[1, 2], 0.0]))

    def _qp_terms(self, t, x0, terminal):
        """q, A, l, u of the MPC QP for initial state x0 (axis, 3) at time t."""
        cfg, M = self.cfg, self.M
        N, T, Zs, Zu = cfg.mpc_N, cfg.mpc_T, self.M["Zs"], self.M["Zu"]
        tk = t + T * np.arange(1, N + 1)
        zref = np.array([self.zmp_ref_c(tt) for tt in tk])                      # (N, 2)
        z_free = np.stack([Zs @ x0[0], Zs @ x0[1]], axis=1)                       # ZMP with U = 0
        q = 2 * cfg.gamma * np.concatenate([Zu.T @ (z_free[:, 0] - zref[:, 0]), Zu.T @ (z_free[:, 1] - zref[:, 1])])
        A = np.zeros((N * MAX_EDGES + 2, 2 * N))
        lo = np.full(N * MAX_EDGES + 2, -np.inf); hi = np.full(N * MAX_EDGES + 2, np.inf)
        for k in range(N):
            nrm, d, _, _ = self.support_at(tk[k])
            r = k * MAX_EDGES
            L = len(d)
            A[r:r + L, :N] = nrm[:, :1] * Zu[k]
            A[r:r + L, N:] = nrm[:, 1:] * Zu[k]
            hi[r:r + L] = d - nrm @ z_free[k]
        if terminal:
            xr = self.xi_ref(tk[-1])
            for a in range(2):
                A[-2 + a, a * N:(a + 1) * N] = self.Xi_u
                lo[-2 + a] = xr[a] - cfg.term_eps - self.Xi_s @ x0[a]
                hi[-2 + a] = xr[a] + cfg.term_eps - self.Xi_s @ x0[a]
        return q, A, lo, hi, zref

    def mpc_solve(self, t, c_xy, cd_xy):
        """One MPC solve from the measured CoM (or from the plan when feedback is off). Updates the plan."""
        cfg = self.cfg
        N = cfg.mpc_N
        x_plan = self.plan_eval(t)
        x0 = x_plan.copy()
        if cfg.feedback:
            x0[:, 0], x0[:, 1] = c_xy, cd_xy                 # c'' stays from the plan (measured c'' is noisy)
        warm = self.plan_U.reshape(-1)
        t_start = time.perf_counter()
        status, U, it = 2, None, 0
        for status_try, terminal in ((0, cfg.use_terminal), (1, False)):
            if status_try == 1 and not cfg.use_terminal:
                break
            q, A, lo, hi, zref = self._qp_terms(t, x0, terminal)
            solver = osqp.OSQP()
            solver.setup(self.P_mpc, q, sp.csc_matrix(A), lo, hi, verbose=False, eps_abs=1e-7, eps_rel=1e-7,
                         polishing=False, max_iter=10000, warm_starting=True)
            solver.warm_start(x=warm)
            res = solver.solve()
            it += int(res.info.iter)
            sv = int(res.info.status_val)
            if (sv in OK or sv in WARN) and res.x is not None and np.all(np.isfinite(res.x)):
                status, U = status_try, res.x
                break
        ms = (time.perf_counter() - t_start) * 1e3
        self.n_status[status] += 1
        self.mpc_ms.append(ms)
        st = self.mpc_state
        st.update(mpc_status=status, mpc_ms=ms, mpc_iter=it)
        if U is None:                                        # keep the previous plan
            return status
        self.plan_t, self.plan_x0, self.plan_U = t, x0, U.reshape(2, N)
        self._plan_states()
        Zs, Zu = self.M["Zs"], self.M["Zu"]
        zp = np.stack([Zs @ x0[a] + Zu @ self.plan_U[a] for a in range(2)], axis=1)
        cp = np.stack([self.M["Ps"] @ x0[a] + self.M["Pu"] @ self.plan_U[a] for a in range(2)], axis=1)
        xiN = np.array([self.Xi_s @ x0[a] + self.Xi_u @ self.plan_U[a] for a in range(2)])
        xrN = self.xi_ref(t + N * cfg.mpc_T)
        Ax = A @ U
        st.update(zmp_pred=zp, com_pred=cp, xi_N=xiN, xi_ref_N=xrN,
                  zmp_active=int(np.sum(Ax[:N * MAX_EDGES] > hi[:N * MAX_EDGES] - 1e-4)),
                  term_active=int(cfg.use_terminal and np.max(np.abs(xiN - xrN)) > cfg.term_eps - 1e-4))
        return status

    def solve(self, t):
        cfg, m, d = self.cfg, self.m, self.d
        if t >= self.next_replan - 1e-9:
            mujoco.mj_forward(m, d)
            Jcom = np.zeros((3, self.nv)); mujoco.mj_jacSubtreeCom(m, d, Jcom, self.PELVIS)
            c, cd = d.subtree_com[self.PELVIS], Jcom @ d.qvel
            self.mpc_solve(t, c[:2].copy(), cd[:2].copy())
            self.next_replan = t + cfg.replan_dt
        tau, info = super().solve(t)                         # walk_core ID-QP with com_ref() from the MPC
        v = self.plan_eval(t)
        xi = info["c"][:2] + info["cd"][:2] / self.omega
        xr = self.xi_ref(t)
        self.mpc_state.update(xi=xi, xi_ref=xr, dcm_err=float(np.linalg.norm(xi - xr)),
                              zmp_mpc=v[:, 0] - self.hg * v[:, 2])
        return tau, info

    # ================================================================== monitor hooks
    def touchdown_info(self, swing_foot, logger):
        n = self.n_status
        return f"MPC solves ok {n[0]}, no-terminal {n[1]}, hold {n[2]}"

    def overlay_rows(self, st):
        e = st.get("dcm_err", np.nan)
        return [("MPC", f"{STATUS[int(st.get('mpc_status', 0))]} {st.get('mpc_ms', 0):.2f} ms"),
                ("ZMP rows @edge", f"{int(st.get('zmp_active', 0))}" + ("  term@edge" if st.get("term_active") else "")),
                ("|xi - xi_ref|", f"{e * 1e3:.1f} mm" if np.isfinite(e) else "-")]

    def extra_markers(self, scn, st):
        """Predicted CoM (blue line), predicted ZMP (green / red at an edge), xi now (red), xi_N vs xi_ref_N."""
        z = 0.004
        zp, cp = st.get("zmp_pred"), st.get("com_pred")
        if zp is None or not np.all(np.isfinite(zp)):
            return
        T, N = self.cfg.mpc_T, self.cfg.mpc_N
        t_plan = self.plan_t
        for k in range(N):
            nrm, d, _, _ = self.support_at(t_plan + (k + 1) * T)
            at_edge = np.max(nrm @ zp[k] - d) > -1e-4
            _add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.007, 0, 0], [zp[k, 0], zp[k, 1], z],
                      [1, 0.1, 0.1, 1] if at_edge else [0.1, 0.9, 0.2, 0.9])
            if k:
                _add_line(scn, [cp[k - 1, 0], cp[k - 1, 1], z], [cp[k, 0], cp[k, 1], z], [0.2, 0.4, 1, 0.9], 0.003)
        xi = st.get("xi")
        if xi is not None and np.all(np.isfinite(xi)):
            _add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.015, 0, 0], [xi[0], xi[1], z], [1, 0.1, 0.1, 1])
        xN, xrN = st["xi_N"], st["xi_ref_N"]
        if np.all(np.isfinite(xN)):
            _add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.012, 0, 0], [xN[0], xN[1], z], [0.6, 0.2, 1, 1])
            e = self.cfg.term_eps
            _add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [e, e, 0.0008], [xrN[0], xrN[1], 0.001], [1, 0.6, 0.7, 0.5])

    def log_extra(self):
        cfg = self.cfg
        return dict(super().log_extra(), omega=self.omega, mpc_T=cfg.mpc_T, mpc_N=cfg.mpc_N, alpha=cfg.alpha,
                    gamma=cfg.gamma, zmp_margin=cfg.zmp_margin, use_terminal=cfg.use_terminal, term_eps=cfg.term_eps,
                    feedback=cfg.feedback, replan_dt=cfg.replan_dt, xi_tg=self.xi_tg, xi_grid=self.xi_grid,
                    mpc_status_counts=np.array([self.n_status[i] for i in range(3)]),
                    mpc_ms_all=np.array(self.mpc_ms))


# ====================================================================== offline check (no robot dynamics)
def offline_check(alphas=(1e-8, 1e-7, 1e-6, 1e-5, 1e-4)):
    """1) condensed matrices vs recursion and vs the closed form of Wieber 2006;
    2) MPC on the ideal LIPM (x0 always from its own plan) vs the preview plan of walk_core, per alpha."""
    ctrl = MPCWalkController(MPCConfig())
    T, N, hg = ctrl.cfg.mpc_T, ctrl.cfg.mpc_N, ctrl.hg
    M = ctrl.M
    rng = np.random.default_rng(0)
    x0, U = rng.normal(size=3), rng.normal(size=N)
    x, z_rec = x0.copy(), []
    for k in range(N):
        x = M["A"] @ x + M["B"] * U[k]; z_rec.append(x[0] - hg * x[2])
    err_rec = np.max(np.abs(M["Zs"] @ x0 + M["Zu"] @ U - np.array(z_rec)))
    i = np.arange(1, N + 1)
    Zu_cf = np.zeros((N, N))
    for k in range(N):
        for j in range(k + 1):
            n = k - j + 1
            Zu_cf[k, j] = (1 - 3 * n + 3 * n * n) * T ** 3 / 6 - T * hg
    Zs_cf = np.stack([np.ones(N), i * T, i * i * T * T / 2 - hg], axis=1)
    print(f"matrices: |Z(recursion) - Z(condensed)| = {err_rec:.1e}, |Zu - closed form| = "
          f"{np.max(np.abs(M['Zu'] - Zu_cf)):.1e}, |Zs - closed form| = {np.max(np.abs(M['Zs'] - Zs_cf)):.1e}")
    print(f"T={T}s N={N} h/g={hg:.4f} | preview plan: dt={ctrl.DTP}s, R/Q={ctrl.cfg.preview_R:g}")
    print(f"{'alpha/gamma':>11} | {'max|c-c_prev|':>13} | {'max|z-z_ref|':>12} | {'max ZMP outside':>15} | "
          f"{'max|u|':>7} | {'fallbacks':>9} | {'mean/max ms':>11}")
    tg = np.arange(0, ctrl.T_TOTAL, 0.002)
    for a in alphas:
        c = MPCWalkController(MPCConfig(alpha=a, feedback=False))
        com, zmp, umax = [], [], 0.0
        for t in tg:
            if t >= c.next_replan - 1e-9:
                c.mpc_solve(t, None, None)
                c.next_replan = t + c.cfg.replan_dt
                umax = max(umax, abs(c.plan_U[:, 0]).max())
            v = c.plan_eval(t)
            com.append(v[:, 0]); zmp.append(v[:, 0] - c.hg * v[:, 2])
        com, zmp = np.array(com), np.array(zmp)
        prev = np.array([c.com_plan[min(int(round(t / c.DTP)), c.NPLAN - 1), 0] for t in tg])
        zref = np.array([c.zmp_ref_c(t) for t in tg])
        outside = 0.0
        for t, p in zip(tg, zmp):          # against the polygon WITHOUT margin, on the 2 ms grid (between samples too)
            hull = c.support_at(t)[3]
            lines, _ = shrink_polygon(hull, 0.0)
            outside = max(outside, max(n @ p - d for n, d in lines))
        print(f"{a:>11.0e} | {np.max(np.linalg.norm(com - prev, axis=1)) * 1e3:>10.1f} mm | "
              f"{np.max(np.linalg.norm(zmp - zref, axis=1)) * 1e3:>9.1f} mm | {max(outside, 0) * 1e3:>12.1f} mm | "
              f"{umax:>7.2f} |{c.n_status[1] + c.n_status[2]:>9} | "
              f"{np.mean(c.mpc_ms):.2f}/{np.max(c.mpc_ms):.2f}")


if __name__ == "__main__":
    offline_check()
