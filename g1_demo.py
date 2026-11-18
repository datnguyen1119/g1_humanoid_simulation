"""
ID-QP (inverse-dynamics QP) giữ thăng bằng cho Unitree G1 trong MuJoCo.

Biến quyết định: x = [qdd (nv), tau (nu), f (8 góc chân x 3)]
  min  sum_i w_i ||A_i x - b_i||^2           (CoM, hướng thân, tư thế, regularize)
  s.t. M qdd + h = S^T tau + Jc^T f          (động lực học, 35 hàng)
       J_foot qdd + dJ_foot qd = -Kf J_foot qd   (2 bàn chân đứng yên, 6D mỗi chân)
       |tau| <= tau_max
       f_z >= 0, |f_x|, |f_y| <= mu f_z      (kim tự tháp ma sát tại mỗi góc chân)
Giải bằng OSQP mỗi bước mô phỏng (500 Hz), gửi tau xuống actuator dạng motor.
"""
import os, time, sys
import numpy as np
import scipy.sparse as sp
import mujoco, osqp

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.environ.get("G1_XML", os.path.join(HERE, "..", "mujoco_menagerie", "unitree_g1", "scene.xml"))
OUT = os.environ.get("OUT_DIR", os.path.join(HERE, "out"))
os.makedirs(OUT, exist_ok=True)
RECORD = os.environ.get("RECORD", "1") == "1"
T_END = float(os.environ.get("T_END", "10.0"))

# ---------------------------------------------------------------- model
m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
d2 = mujoco.MjData(m)  # bản nháp để tính dJ*qd bằng sai phân
nq, nv, nu = m.nq, m.nv, m.nu
dt = m.opt.timestep

# Đổi actuator position (PD nội bộ) -> motor (torque thuần): tau = ctrl
tau_max = m.actuator_forcerange[:, 1].copy()
m.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
m.actuator_gainprm[:] = 0; m.actuator_gainprm[:, 0] = 1.0
m.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_NONE
m.actuator_biasprm[:] = 0
m.actuator_ctrllimited[:] = 1
m.actuator_ctrlrange[:, 0] = -tau_max; m.actuator_ctrlrange[:, 1] = tau_max
# map actuator -> dof (với G1 là 1-1, theo thứ tự khớp)
act_dof = np.array([m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in range(nu)])
S = np.zeros((nu, nv)); S[np.arange(nu), act_dof] = 1.0

def jid(name): return m.joint(name).qposadr[0]
def bid(name): return m.body(name).id

# ---------------------------------------------------------------- tư thế ban đầu (gối hơi co)
q_nom = m.key_qpos[0].copy()
for side in ["left", "right"]:
    q_nom[jid(f"{side}_hip_pitch_joint")] = -0.312
    q_nom[jid(f"{side}_knee_joint")] = 0.669
    q_nom[jid(f"{side}_ankle_pitch_joint")] = -0.357
d.qpos[:] = q_nom; mujoco.mj_kinematics(m, d)
foot_geoms = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
              and m.body(m.geom_bodyid[g]).name.endswith("ankle_roll_link")]
zmin = min(d.geom_xpos[g][2] - m.geom_size[g, 0] for g in foot_geoms) #[2] = z của tâm quả cầu - bán kính = z chạm sàn
q_nom[2] -= zmin - 1e-4         # đặt chân vừa chạm sàn
d.qpos[:] = q_nom; d.qvel[:] = 0
mujoco.mj_forward(m, d)

FOOT_BODIES = [bid("left_ankle_roll_link"), bid("right_ankle_roll_link")]
TORSO = bid("torso_link")
# 4 góc mỗi chân (toạ độ tâm quả cầu va chạm trong frame bàn chân, trừ bán kính theo z)
corners = []  # (body, local_pos)
for g in foot_geoms:
    p = m.geom_pos[g].copy(); p[2] -= m.geom_size[g, 0]
    corners.append((m.geom_bodyid[g], p))
NC = len(corners)          # 8
nf = 3 * NC                # 24 biến lực, mỗi điểm có 3 thành phần lực
NX = nv + nu + nf

# ---------------------------------------------------------------- jacobian helpers
_jp = np.zeros((3, nv)); _jr = np.zeros((3, nv))

def task_jacobians(data):
    """Trả về các Jacobian cần đạo hàm: feet6D(12xnv), com(3xnv), torso_rot(3xnv)."""
    Jf = []
    for b in FOOT_BODIES:
        mujoco.mj_jacBody(m, data, _jp, _jr, b)
        Jf += [_jp.copy(), _jr.copy()]
    Jcom = np.zeros((3, nv)); mujoco.mj_jacSubtreeCom(m, data, Jcom, 1)
    mujoco.mj_jacBody(m, data, _jp, _jr, TORSO)
    return np.vstack(Jf), Jcom, _jr.copy()

def dJqd(q, qd, eps=1e-6):
    """J̇ q̇ ≈ (J(q + eps q̇) - J(q))/eps · q̇  (sai phân theo hướng q̇)."""
    d2.qpos[:] = q; mujoco.mj_kinematics(m, d2); mujoco.mj_comPos(m, d2) # trong mj_comPos có tính subtree_com rồi
    J0 = task_jacobians(d2)
    d2.qpos[:] = q; mujoco.mj_integratePos(m, d2.qpos, qd, eps)
    mujoco.mj_kinematics(m, d2); mujoco.mj_comPos(m, d2)
    J1 = task_jacobians(d2)
    return [((b - a) / eps) @ qd for a, b in zip(J0, J1)]

def corner_jac(data):
    Jc = np.zeros((nf, nv))
    for i, (b, p) in enumerate(corners):
        pw = data.xpos[b] + data.xmat[b].reshape(3, 3) @ p
        mujoco.mj_jac(m, data, _jp, None, pw, b)
        Jc[3*i:3*i+3] = _jp
    return Jc

# ---------------------------------------------------------------- tham chiếu CoM
c0 = d.subtree_com[1].copy()
def smooth(s):  # 0->1 mượt (vận tốc, gia tốc bằng 0 ở hai đầu)
    s = np.clip(s, 0, 1); return s*s*s*(10 - 15*s + 6*s*s)
def com_ref_pos(t):
    c = c0.copy()
    # 1.0-2.5 s dịch CoM sang trái 5 cm, 3.0-4.0 s về lại
    c[1] += 0.05 * (smooth((t - 1.0) / 1.5) - smooth((t - 3.0) / 1.0))
    # 4.0-6.0 s ngồi xuống 6 cm rồi đứng lên
    c[2] -= 0.06 * (smooth((t - 4.0) / 1.0) - smooth((t - 5.0) / 1.0))
    return c
def com_ref(t, h=1e-3):
    p = com_ref_pos(t); pp = com_ref_pos(t + h); pm = com_ref_pos(t - h)
    return p, (pp - pm) / (2*h), (pp - 2*p + pm) / (h*h)

# Đẩy: (t_start, duration, force_world)
PUSHES = [(6.5, 0.2, np.array([60.0, 0, 0])),    # đẩy về phía trước
          (8.0, 0.2, np.array([0, -60.0, 0]))]   # đẩy ngang sang phải
def push_force(t):
    for t0, T, F in PUSHES:
        if t0 <= t < t0 + T: return F
    return np.zeros(3)

# ---------------------------------------------------------------- gains & weights
KP_COM, KD_COM = 100.0, 20.0
KP_ROT, KD_ROT = 200.0, 30.0
KP_POST, KD_POST = 60.0, 15.0
KF = 20.0                     # Baumgarte cho ràng buộc chân
W_COM, W_ROT, W_POST, W_POST_LEG = 100.0, 20.0, 1.0, 0.05
W_TAU, W_F, W_QDD = 1e-4, 1e-5, 1e-6
MU = 0.5                      # (sàn MuJoCo dùng 0.6, lấy nhỏ hơn cho an toàn)

iq, it, iff = slice(0, nv), slice(nv, nv+nu), slice(nv+nu, NX)

# Ràng buộc ma sát cố định (không đổi theo thời gian)
fr_rows = []
for i in range(NC):
    fx, fy, fz = nv+nu+3*i, nv+nu+3*i+1, nv+nu+3*i+2
    for a, b in [(fx, 1), (fx, -1), (fy, 1), (fy, -1)]:
        r = np.zeros(NX); r[a] = b; r[fz] = -MU; fr_rows.append(r)   # ±f_t - mu f_z <= 0
    r = np.zeros(NX); r[fz] = 1; fr_rows.append(r)                    # 0 <= f_z <= 600
A_fric = np.array(fr_rows)
l_fric = np.tile([-np.inf]*4 + [0.0], NC); u_fric = np.tile([0.0]*4 + [600.0], NC)
A_tau = np.zeros((nu, NX)); A_tau[:, it] = np.eye(nu)
# Giới hạn khớp -> chặn qdd sao cho q(t+H) không vượt giới hạn (thêm biên an toàn).
# Gối giữ >= 0.25 rad để tránh duỗi thẳng (điểm kỳ dị, mất khả năng hạ CoM).
jlo = np.array([m.jnt_range[m.actuator_trnid[i, 0], 0] for i in range(nu)]) + 0.05
jhi = np.array([m.jnt_range[m.actuator_trnid[i, 0], 1] for i in range(nu)]) - 0.05
for side in ["left", "right"]:
    jlo[m.actuator(f"{side}_knee_joint").id] = 0.25
A_lim = np.zeros((nu, NX)); A_lim[:, iq][:, :] = 0; A_lim[np.arange(nu), act_dof] = 1.0 #ma trận dùng để nhân với x và lấy gia tốc
H_LIM = 0.08

def solve_idqp(t):
    q, qd = d.qpos.copy(), d.qvel.copy()
    M = np.zeros((nv, nv)); mujoco.mj_fullM(m, d, M)
    h = d.qfrc_bias.copy()                      # C(q,qd) qd + g(q)
    Jf, Jcom, Jrot = task_jacobians(d)
    dJf, dJcom, dJrot = dJqd(q, qd)
    Jc = corner_jac(d)

    # --- tasks: ||A x - b||^2
    tasks = []
    c, cd = d.subtree_com[1].copy(), Jcom @ qd
    cr, cdr, cddr = com_ref(t)
    a_com = cddr + KP_COM*(cr - c) + KD_COM*(cdr - cd)
    A = np.zeros((3, NX)); A[:, iq] = Jcom; tasks.append((W_COM, A, a_com - dJcom))

    quat = d.xquat[TORSO]; err = np.zeros(3)
    mujoco.mju_subQuat(err, np.array([1.0, 0, 0, 0]), quat)   # lỗi hướng (local frame)
    err = d.xmat[TORSO].reshape(3, 3) @ err                   # -> world frame
    a_rot = KP_ROT*err - KD_ROT*(Jrot @ qd)
    A = np.zeros((3, NX)); A[:, iq] = Jrot; tasks.append((W_ROT, A, a_rot - dJrot))

    # Tư thế: chân chỉ giữ nhẹ (để CoM task được phép co gối), thân trên + tay giữ chặt hơn
    a_post = KP_POST*(q_nom[7:] - q[7:]) - KD_POST*qd[6:]
    wj = np.sqrt(np.where(np.arange(nu) < 12, W_POST_LEG, W_POST))
    A = np.zeros((nv-6, NX)); A[:, 6:nv] = np.diag(wj); tasks.append((1.0, A, wj*a_post))

    P = np.zeros((NX, NX)); g = np.zeros(NX)
    for w, A, b in tasks:
        P += 2*w*A.T@A; g -= 2*w*A.T@b
    P[iq, iq] += 2*W_QDD*np.eye(nv); P[it, it] += 2*W_TAU*np.eye(nu); P[iff, iff] += 2*W_F*np.eye(nf)

    # --- constraints
    A_dyn = np.hstack([M, -S.T, -Jc.T]); b_dyn = -h
    A_feet = np.hstack([Jf, np.zeros((12, nu+nf))]); b_feet = -dJf - KF*(Jf @ qd)
    qj, vj = q[7:], qd[6:]
    qdd_lo = 2*(jlo - qj - vj*H_LIM)/H_LIM**2; qdd_hi = 2*(jhi - qj - vj*H_LIM)/H_LIM**2
    qdd_lo = np.minimum(qdd_lo, qdd_hi)
    Acons = np.vstack([A_dyn, A_feet, A_tau, A_fric, A_lim])
    lo = np.concatenate([b_dyn, b_feet, -tau_max, l_fric, qdd_lo])
    hi = np.concatenate([b_dyn, b_feet,  tau_max, u_fric, np.maximum(qdd_hi, qdd_lo)])

    solver = osqp.OSQP()
    solver.setup(sp.csc_matrix(np.triu(P)), g, sp.csc_matrix(Acons), lo, hi,
                 verbose=False, eps_abs=1e-6, eps_rel=1e-6, polishing=True, max_iter=20000)
    res = solver.solve()
    x = res.x
    return x, res.info.status, c, cr, Jc

# ---------------------------------------------------------------- simulation loop
renderer = None
if RECORD:
    renderer = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera(); cam.distance, cam.azimuth, cam.elevation = 2.2, 140, -12
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
    m.vis.map.force = 0.003
    frames, FPS = [], 30
    from PIL import Image, ImageDraw, ImageFont
    try:
        FONT = ImageFont.truetype("DejaVuSans.ttf", 22); FONT_S = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        FONT = FONT_S = ImageFont.load_default()

log = {k: [] for k in ["qpos", "feet", "t", "com", "com_ref", "tau", "fz_qp", "fz_sim", "status", "solve_ms", "push"]}
lfoot = set(FOOT_BODIES[:1]); rfoot = set(FOOT_BODIES[1:])
cf = np.zeros(6)
step = 0; fell = False
while d.time < T_END:
    t = d.time
    t0 = time.perf_counter()
    x, status, c, cr, Jc = solve_idqp(t)
    ms = (time.perf_counter() - t0) * 1e3
    if x is None or not np.all(np.isfinite(x)):
        print(f"QP failed at t={t:.3f}: {status}"); break
    tau = x[nv:nv+nu]
    d.ctrl[:] = tau
    F = push_force(t)
    d.xfrc_applied[:] = 0; d.xfrc_applied[TORSO, :3] = F

    # log
    f = x[nv+nu:].reshape(NC, 3)
    fz_qp = [f[[i for i, (b, _) in enumerate(corners) if b == fb], 2].sum() for fb in FOOT_BODIES]
    fz_sim = [0.0, 0.0]
    for k in range(d.ncon):
        con = d.contact[k]
        bodies = {m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]}
        mujoco.mj_contactForce(m, d, k, cf)
        fn = cf[0]
        if bodies & lfoot: fz_sim[0] += fn
        elif bodies & rfoot: fz_sim[1] += fn
    feet = np.concatenate([d.xpos[b] for b in FOOT_BODIES])
    for k, v in zip(log, [q.copy() if False else d.qpos.copy(), feet, t, c, cr, tau.copy(), fz_qp, fz_sim, status, ms, F.copy()]):
        log[k].append(v)

    if RECORD and step % int(round(1/(FPS*dt))) == 0:
        cam.lookat[:] = [c[0], c[1], 0.6]
        renderer.update_scene(d, camera=cam, scene_option=opt)
        img = Image.fromarray(renderer.render().copy()); dr = ImageDraw.Draw(img)
        ph = ("đứng yên" if t < 1 else "dịch CoM sang trái 5 cm" if t < 4 else "ngồi xuống 6 cm rồi đứng lên" if t < 6.5
              else "ĐẨY về trước 60 N x 0.2 s" if t < 6.7 else "hồi phục" if t < 8 else "ĐẨY ngang 60 N x 0.2 s" if t < 8.2 else "hồi phục")
        dr.text((12, 10), f"t = {t:4.2f} s   {ph}", fill=(255, 80, 80) if F.any() else (255, 255, 255), font=FONT)
        dr.text((12, 40), f"ID-QP (OSQP) {ms:4.1f} ms  |  lỗi CoM {np.linalg.norm(c-cr)*1e3:5.1f} mm", fill=(220, 220, 220), font=FONT_S)
        frames.append(np.asarray(img))

    mujoco.mj_step(m, d)
    step += 1
    if d.qpos[2] < 0.4:
        print(f"Robot ngã tại t={d.time:.2f}s"); fell = True; break
    if step % 500 == 0:
        print(f"t={d.time:.2f}s  com_err={np.linalg.norm(c-cr)*1e3:.1f}mm  qp={ms:.1f}ms  {status}", flush=True)

np.savez(os.path.join(OUT, "log.npz"), **{k: np.array(v) for k, v in log.items() if k != "status"},
         status=np.array(log["status"]), pushes=np.array([[p[0], p[1], *p[2]] for p in PUSHES]))
if RECORD and frames:
    import imageio
    imageio.mimsave(os.path.join(OUT, "g1_idqp_balance.mp4"), frames, fps=1/(int(round(1/(FPS*dt)))*dt), quality=8)
print("fell" if fell else "OK, đứng vững tới hết", "| status khác solved:",
      sum(s != "solved" for s in log["status"]), "/", len(log["status"]),
      f"| QP trung bình {np.mean(log['solve_ms']):.2f} ms")
