"""
G1 đi bộ: lịch bước chân -> ZMP tham chiếu -> ZMP preview control (Kajita 2003, LIPM)
-> quỹ đạo CoM + quỹ đạo chân bay -> ID-QP (OSQP) đổi tiếp xúc theo pha.

x = [qdd (nv), tau (nu), f (8 góc chân x 3)]
  min  w_com||a_com err||² + w_sw||a_swing err||² + w_rot||thân|| + tư thế + regularize
  s.t. M qdd + h = S^T tau + Jc^T f
       chân trụ đứng yên (6D), chân bay: lực = 0
       ma sát, giới hạn torque, giới hạn khớp
"""
import os, time
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
import mujoco, osqp

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.environ.get("G1_XML", os.path.join(HERE, "..", "mujoco_menagerie", "unitree_g1", "scene.xml"))
OUT = os.environ.get("OUT_DIR", os.path.join(HERE, "out_walk"))
os.makedirs(OUT, exist_ok=True)
RECORD = os.environ.get("RECORD", "1") == "1"
N_STEPS = int(os.environ.get("N_STEPS", "8"))
STEP_LEN = float(os.environ.get("STEP_LEN", "0.10"))
T_SS = float(os.environ.get("T_SS", "0.5"))
T_DS = float(os.environ.get("T_DS", "0.2"))
STEP_H = float(os.environ.get("STEP_H", "0.05"))
PUSH = os.environ.get("PUSH", "0") == "1"

# ================================================================ model (giống bản cân bằng)
m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m); d2 = mujoco.MjData(m)
nq, nv, nu = m.nq, m.nv, m.nu
dt = m.opt.timestep
G = 9.81

tau_max = m.actuator_forcerange[:, 1].copy()
m.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
m.actuator_gainprm[:] = 0; m.actuator_gainprm[:, 0] = 1.0
m.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_NONE
m.actuator_biasprm[:] = 0
m.actuator_ctrllimited[:] = 1
m.actuator_ctrlrange[:, 0] = -tau_max; m.actuator_ctrlrange[:, 1] = tau_max
act_dof = np.array([m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in range(nu)])
S = np.zeros((nu, nv)); S[np.arange(nu), act_dof] = 1.0

def jid(name): return m.joint(name).qposadr[0]
def bid(name): return m.body(name).id

q_nom = m.key_qpos[0].copy()
for side in ["left", "right"]:
    q_nom[jid(f"{side}_hip_pitch_joint")] = -0.312
    q_nom[jid(f"{side}_knee_joint")] = 0.669
    q_nom[jid(f"{side}_ankle_pitch_joint")] = -0.357
d.qpos[:] = q_nom; mujoco.mj_kinematics(m, d)
foot_geoms = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
              and m.body(m.geom_bodyid[g]).name.endswith("ankle_roll_link")]
zmin = min(d.geom_xpos[g][2] - m.geom_size[g, 0] for g in foot_geoms)
q_nom[2] -= zmin - 1e-4
d.qpos[:] = q_nom; d.qvel[:] = 0
mujoco.mj_forward(m, d)

FEET = [bid("left_ankle_roll_link"), bid("right_ankle_roll_link")]   # 0 = trái, 1 = phải
TORSO, PELVIS = bid("torso_link"), bid("pelvis")
corners = []
for g in foot_geoms:
    p = m.geom_pos[g].copy(); p[2] -= m.geom_size[g, 0]
    corners.append((m.geom_bodyid[g], p))
NC = len(corners); nf = 3*NC; NX = nv + nu + nf
corner_foot = np.array([FEET.index(b) for b, _ in corners])
iq, it, iff = slice(0, nv), slice(nv, nv+nu), slice(nv+nu, NX)
SOLE_OFF = np.array([0.035, 0.0])      # tâm bàn chân so với gốc frame cổ chân (theo x)

# ================================================================ 1. lịch bước chân
foot0 = [d.xpos[b].copy() for b in FEET]
c0 = d.subtree_com[1].copy()
ZC = c0[2]
T_INIT, T_END_STAND = 1.0, 1.5
# mỗi bước: (chân bay, vị trí đích của gốc cổ chân)
steps = []
pos = [foot0[0].copy(), foot0[1].copy()]
swing = 0                                   # bắt đầu bằng chân trái
for k in range(N_STEPS):
    tgt = pos[swing].copy()
    if k == N_STEPS - 1:
        tgt[0] = pos[1 - swing][0]          # bước cuối: khép chân
    else:
        tgt[0] = pos[1 - swing][0] + STEP_LEN   # STEP_LEN = khoảng cách 2 chân theo x
    steps.append((swing, pos[swing].copy(), tgt.copy()))
    pos[swing] = tgt
    swing = 1 - swing

# pha: (t0, t1, kind, info). kind in {"stand","ds","ss"}
phases = [(0.0, T_INIT, "stand", None)]
t = T_INIT
zmp_pts = [c0[:2].copy()]                   # điểm ZMP "neo" sau mỗi DS
cur_feet = [foot0[0][:2].copy(), foot0[1][:2].copy()]
for k, (sw, p_from, p_to) in enumerate(steps):
    stance = 1 - sw
    z_to = cur_feet[stance] + SOLE_OFF
    phases.append((t, t + T_DS, "ds", (zmp_pts[-1].copy(), z_to.copy()))); t += T_DS
    phases.append((t, t + T_SS, "ss", (sw, p_from, p_to, z_to.copy()))); t += T_SS
    zmp_pts.append(z_to); cur_feet[sw] = p_to[:2].copy()
z_mid = (cur_feet[0] + cur_feet[1]) / 2 + SOLE_OFF
z_mid[0] = (cur_feet[0][0] + cur_feet[1][0]) / 2 + (c0[0] - foot0[0][0])   # giữ offset CoM ban đầu
phases.append((t, t + 0.6, "ds", (zmp_pts[-1].copy(), z_mid.copy()))); t += 0.6
phases.append((t, t + T_END_STAND, "stand_end", z_mid.copy())); t += T_END_STAND
T_TOTAL = t

def smooth(s):
    s = np.clip(s, 0, 1); return s*s*s*(10 - 15*s + 6*s*s)

def phase_at(t):
    for ph in phases:
        if ph[0] <= t < ph[1]: return ph
    return phases[-1]

def zmp_ref(t):
    t0, t1, kind, info = phase_at(t)
    if kind == "stand": return c0[:2].copy()
    if kind == "stand_end": return info.copy()
    if kind == "ds":
        a, b = info; return a + (b - a) * smooth((t - t0) / (t1 - t0))
    return info[3].copy()

# ================================================================ 2. ZMP preview control (Kajita)
DTP = 0.005
NPREV = int(1.6 / DTP)
A = np.array([[1, DTP, DTP**2/2], [0, 1, DTP], [0, 0, 1]])
B = np.array([[DTP**3/6], [DTP**2/2], [DTP]])
C = np.array([[1, 0, -ZC/G]])
At = np.block([[np.eye(1), C @ A], [np.zeros((3, 1)), A]])
Bt = np.vstack([C @ B, B])
Qt = np.diag([1.0, 0, 0, 0]); R = np.array([[1e-6]])
P = sla.solve_discrete_are(At, Bt, Qt, R)
Kt = np.linalg.solve(R + Bt.T @ P @ Bt, Bt.T @ P @ At)
Gi, Gx = Kt[0, 0], Kt[0, 1:]
Ac = At - Bt @ Kt
It = np.array([[1.0], [0], [0], [0]])
X = -Ac.T @ P @ It
Gd = np.zeros(NPREV); Gd[0] = -Gi
for j in range(1, NPREV):
    Gd[j] = (np.linalg.solve(R + Bt.T @ P @ Bt, Bt.T @ X))[0, 0]
    X = Ac.T @ X

tp = np.arange(0, T_TOTAL + NPREV*DTP + DTP, DTP)
pref = np.array([zmp_ref(min(ti, T_TOTAL - 1e-6)) for ti in tp])     # (N,2)
com_plan = np.zeros((len(tp), 3, 2))   # [k, (pos,vel,acc), axis]
zmp_plan = np.zeros((len(tp), 2))
for ax in range(2):
    x = np.array([c0[ax], 0, 0]); esum = 0.0
    for k in range(len(tp) - NPREV):
        pz = (C @ x)[0]; esum += pz - pref[k, ax]
        u = -Gi*esum - Gx @ x - Gd[1:] @ pref[k+1:k+NPREV, ax]
        com_plan[k, :, ax] = x; zmp_plan[k, ax] = pz
        x = A @ x + B[:, 0]*u
NPLAN = len(tp) - NPREV

def com_ref(t):
    k = min(int(t / DTP), NPLAN - 2); a = t/DTP - k
    v = (1-a)*com_plan[k] + a*com_plan[k+1]
    p = np.array([v[0, 0], v[0, 1], ZC]); vel = np.array([v[1, 0], v[1, 1], 0]); acc = np.array([v[2, 0], v[2, 1], 0])
    return p, vel, acc

# ================================================================ 3. quỹ đạo chân bay
def swing_pos(p_from, p_to, s):
    s = np.clip(s, 0, 1)
    p = p_from + (p_to - p_from) * smooth(s)
    p = p.copy(); p[2] = p_from[2] + STEP_H * 16 * s**2 * (1-s)**2   # nhấc nhanh khỏi sàn, tránh kéo lê
    return p
def swing_ref(info, t0, t1, t, h=1e-3):
    sw, a, b = info[:3]; T = t1 - t0
    f = lambda tt: swing_pos(a, b, (tt - t0) / T)
    p, pp, pm = f(t), f(t+h), f(t-h)
    return p, (pp - pm)/(2*h), (pp - 2*p + pm)/(h*h)

# ================================================================ jacobians
_jp = np.zeros((3, nv)); _jr = np.zeros((3, nv))
def task_jacobians(data):
    J = []
    for b in FEET:
        mujoco.mj_jacBody(m, data, _jp, _jr, b); J.append(np.vstack([_jp, _jr]))
    Jcom = np.zeros((3, nv)); mujoco.mj_jacSubtreeCom(m, data, Jcom, 1)
    mujoco.mj_jacBody(m, data, _jp, _jr, TORSO); Jt = _jr.copy()
    mujoco.mj_jacBody(m, data, _jp, _jr, PELVIS); Jpv = _jr.copy()
    return J[0], J[1], Jcom, Jt, Jpv
def dJqd(q, qd, eps=1e-6):
    d2.qpos[:] = q; mujoco.mj_kinematics(m, d2); mujoco.mj_comPos(m, d2); J0 = task_jacobians(d2)
    d2.qpos[:] = q; mujoco.mj_integratePos(m, d2.qpos, qd, eps)
    mujoco.mj_kinematics(m, d2); mujoco.mj_comPos(m, d2); J1 = task_jacobians(d2)
    return [((b - a)/eps) @ qd for a, b in zip(J0, J1)]
def corner_jac(data):
    Jc = np.zeros((nf, nv))
    for i, (b, p) in enumerate(corners):
        pw = data.xpos[b] + data.xmat[b].reshape(3, 3) @ p
        mujoco.mj_jac(m, data, _jp, None, pw, b); Jc[3*i:3*i+3] = _jp
    return Jc
def orient_err(body, target=np.array([1.0, 0, 0, 0])):
    e = np.zeros(3); mujoco.mju_subQuat(e, target, d.xquat[body])
    return d.xmat[body].reshape(3, 3) @ e

# ================================================================ gains
KP_COM, KD_COM = 100.0, 20.0
KP_ROT, KD_ROT = 200.0, 30.0
KP_SW, KD_SW = 400.0, 40.0
KP_POST, KD_POST = 60.0, 15.0
KF = 20.0
W_COM, W_ROT, W_PELV, W_SW, W_POST, W_POST_LEG = 100.0, 20.0, 10.0, 100.0, 1.0, 0.05
W_TAU, W_F, W_QDD = 1e-4, 1e-5, 1e-6
MU = 0.5

jlo = np.array([m.jnt_range[m.actuator_trnid[i, 0], 0] for i in range(nu)]) + 0.05
jhi = np.array([m.jnt_range[m.actuator_trnid[i, 0], 1] for i in range(nu)]) - 0.05
for side in ["left", "right"]:
    jlo[m.actuator(f"{side}_knee_joint").id] = 0.25
A_lim = np.zeros((nu, NX)); A_lim[np.arange(nu), act_dof] = 1.0
H_LIM = 0.08
A_tau = np.zeros((nu, NX)); A_tau[:, it] = np.eye(nu)
fr_rows = []
for i in range(NC):
    fx, fy, fz = nv+nu+3*i, nv+nu+3*i+1, nv+nu+3*i+2
    for a, b in [(fx, 1), (fx, -1), (fy, 1), (fy, -1)]:
        r = np.zeros(NX); r[a] = b; r[fz] = -MU; fr_rows.append(r)
    r = np.zeros(NX); r[fz] = 1; fr_rows.append(r)
A_fric = np.array(fr_rows)

def solve_idqp(t):
    q, qd = d.qpos.copy(), d.qvel.copy()
    M = np.zeros((nv, nv)); mujoco.mj_fullM(m, d, M)
    h = d.qfrc_bias.copy()
    JL, JR, Jcom, Jt, Jpv = task_jacobians(d)
    dJL, dJR, dJcom, dJt, dJpv = dJqd(q, qd)
    Jfeet, dJfeet = [JL, JR], [dJL, dJR]
    Jc = corner_jac(d)

    t0, t1, kind, info = phase_at(t)
    contact = [True, True]
    if kind == "ss": contact[info[0]] = False

    tasks = []
    c, cd = d.subtree_com[1].copy(), Jcom @ qd
    cr, cdr, cddr = com_ref(t)
    a_com = cddr + KP_COM*(cr - c) + KD_COM*(cdr - cd)
    Ax = np.zeros((3, NX)); Ax[:, iq] = Jcom; tasks.append((W_COM, Ax, a_com - dJcom))
    a_rot = KP_ROT*orient_err(TORSO) - KD_ROT*(Jt @ qd)
    Ax = np.zeros((3, NX)); Ax[:, iq] = Jt; tasks.append((W_ROT, Ax, a_rot - dJt))
    a_pv = KP_ROT*orient_err(PELVIS) - KD_ROT*(Jpv @ qd)
    Ax = np.zeros((3, NX)); Ax[:, iq] = Jpv; tasks.append((W_PELV, Ax, a_pv - dJpv))
    sw_ref = None
    if kind == "ss":
        sw = info[0]; J6, dJ6 = Jfeet[sw], dJfeet[sw]
        pr, vr, ar = swing_ref(info, t0, t1, t); sw_ref = pr
        p, v = d.xpos[FEET[sw]], J6[:3] @ qd
        a_lin = ar + KP_SW*(pr - p) + KD_SW*(vr - v)
        a_ang = KP_SW*orient_err(FEET[sw]) - KD_SW*(J6[3:] @ qd)
        Ax = np.zeros((6, NX)); Ax[:, iq] = J6
        tasks.append((W_SW, Ax, np.concatenate([a_lin, a_ang]) - dJ6))
    a_post = KP_POST*(q_nom[7:] - q[7:]) - KD_POST*qd[6:]
    wj = np.sqrt(np.where(np.arange(nu) < 12, W_POST_LEG, W_POST))
    Ax = np.zeros((nv-6, NX)); Ax[:, 6:nv] = np.diag(wj); tasks.append((1.0, Ax, wj*a_post))

    P = np.zeros((NX, NX)); g = np.zeros(NX)
    for w, Ax, b in tasks:
        P += 2*w*Ax.T@Ax; g -= 2*w*Ax.T@b
    P[iq, iq] += 2*W_QDD*np.eye(nv); P[it, it] += 2*W_TAU*np.eye(nu); P[iff, iff] += 2*W_F*np.eye(nf)

    A_dyn = np.hstack([M, -S.T, -Jc.T]); b_dyn = -h
    rows, lo, hi = [A_dyn], [b_dyn], [b_dyn]
    for f in range(2):
        if contact[f]:
            Af = np.hstack([Jfeet[f], np.zeros((6, nu+nf))]); bf = -dJfeet[f] - KF*(Jfeet[f] @ qd)
            rows.append(Af); lo.append(bf); hi.append(bf)
    fz_hi = np.where(np.array([contact[corner_foot[i]] for i in range(NC)]), 600.0, 0.0)
    l_fric = np.tile([-np.inf]*4 + [0.0], NC); u_fric = np.zeros(5*NC)
    u_fric[4::5] = fz_hi
    qj, vj = q[7:], qd[6:]
    qlo = 2*(jlo - qj - vj*H_LIM)/H_LIM**2; qhi = 2*(jhi - qj - vj*H_LIM)/H_LIM**2
    qlo = np.minimum(qlo, qhi)
    rows += [A_tau, A_fric, A_lim]; lo += [-tau_max, l_fric, qlo]; hi += [tau_max, u_fric, np.maximum(qhi, qlo)]

    solver = osqp.OSQP()
    solver.setup(sp.csc_matrix(np.triu(P)), g, sp.csc_matrix(np.vstack(rows)),
                 np.concatenate(lo), np.concatenate(hi),
                 verbose=False, eps_abs=1e-6, eps_rel=1e-6, polishing=True, max_iter=20000)
    res = solver.solve()
    return res.x, res.info.status, c, cr, kind, sw_ref

# ================================================================ sim
PUSHES = [(T_INIT + 3*(T_DS+T_SS) + 0.3, 0.15, np.array([0, float(os.environ.get("PUSH_F", "40")), 0]))] if PUSH else []
def push_force(t):
    for t0, T, F in PUSHES:
        if t0 <= t < t0 + T: return F
    return np.zeros(3)

if RECORD:
    from PIL import Image, ImageDraw, ImageFont
    renderer = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera(); cam.distance, cam.azimuth, cam.elevation = 2.4, 120, -12
    opt = mujoco.MjvOption(); opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    m.vis.map.force = 0.003
    try: FONT = ImageFont.truetype("DejaVuSans.ttf", 22); FONT_S = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError: FONT = FONT_S = ImageFont.load_default()
    frames, FPS = [], 30
    stride = int(round(1/(FPS*dt)))

log = {k: [] for k in ["t", "com", "com_ref", "zmp_ref", "zmp_meas", "feet", "sw_ref", "fz", "tau", "solve_ms", "qpos"]}
cf = np.zeros(6); step = 0; fell = False; nfail = 0
print(f"T_TOTAL={T_TOTAL:.2f}s, {N_STEPS} bước, zc={ZC:.3f}")
while d.time < T_TOTAL:
    t = d.time
    t_s = time.perf_counter()
    x, status, c, cr, kind, sw_ref = solve_idqp(t)
    ms = (time.perf_counter() - t_s)*1e3
    if status != "solved": nfail += 1
    if x is None or not np.all(np.isfinite(x)):
        print(f"QP lỗi t={t:.3f}: {status}"); break
    d.ctrl[:] = x[nv:nv+nu]
    F = push_force(t); d.xfrc_applied[:] = 0; d.xfrc_applied[TORSO, :3] = F

    # ZMP đo từ lực tiếp xúc MuJoCo (CoP tổng trên mặt sàn)
    fsum, msum, fz = 0.0, np.zeros(2), [0.0, 0.0]
    for k in range(d.ncon):
        con = d.contact[k]
        b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
        if b1 not in FEET and b2 not in FEET: continue
        mujoco.mj_contactForce(m, d, k, cf)
        fn = cf[0]; fsum += fn; msum += fn*con.pos[:2]
        fz[FEET.index(b1 if b1 in FEET else b2)] += fn
    zm = msum/fsum if fsum > 1 else np.full(2, np.nan)
    for k, v in zip(log, [t, c, cr, zmp_ref(t), zm, np.concatenate([d.xpos[b] for b in FEET]),
                          sw_ref if sw_ref is not None else np.full(3, np.nan), fz, x[nv:nv+nu].copy(), ms, d.qpos.copy()]):
        log[k].append(v)

    if RECORD and step % stride == 0:
        cam.lookat[:] = [c[0], c[1], 0.55]
        renderer.update_scene(d, camera=cam, scene_option=opt)
        img = Image.fromarray(renderer.render().copy()); dr = ImageDraw.Draw(img)
        nm = {"stand": "đứng", "stand_end": "dừng", "ds": "đứng 2 chân (dồn ZMP)",
              "ss": "đứng 1 chân, bay chân " + ("trái" if kind == "ss" and phase_at(t)[3][0] == 0 else "phải")}[kind]
        dr.text((12, 10), f"t = {t:4.2f} s   {nm}", fill=(255, 80, 80) if F.any() else (255, 255, 255), font=FONT)
        dr.text((12, 40), f"ID-QP {ms:4.1f} ms | lỗi CoM {np.linalg.norm(c-cr)*1e3:5.1f} mm | x = {c[0]:.2f} m",
                fill=(220, 220, 220), font=FONT_S)
        frames.append(np.asarray(img))

    mujoco.mj_step(m, d); step += 1
    if d.qpos[2] < 0.45:
        print(f"Robot ngã tại t={d.time:.2f}s"); fell = True; break
    if step % 250 == 0:
        print(f"t={d.time:.2f}s {kind:9s} com_err={np.linalg.norm(c-cr)*1e3:5.1f}mm qp={ms:.1f}ms {status}", flush=True)

np.savez(os.path.join(OUT, "log.npz"), **{k: np.array(v) for k, v in log.items()},
         zmp_plan=zmp_plan[:NPLAN], com_plan=com_plan[:NPLAN], tp=tp[:NPLAN],
         phases=np.array([[p[0], p[1], ["stand", "ds", "ss", "stand_end"].index(p[2])] for p in phases]),
         pushes=np.array([[p[0], p[1], *p[2]] for p in PUSHES]).reshape(-1, 5))
if RECORD and frames:
    import imageio
    imageio.mimsave(os.path.join(OUT, "g1_walk.mp4"), frames, fps=1/(stride*dt), quality=8)
print("NGÃ" if fell else "ĐI XONG", f"| x cuối = {d.subtree_com[1][0]:.3f} m | QP không solved: {nfail}"
      f" | QP trung bình {np.mean(log['solve_ms']):.2f} ms")
