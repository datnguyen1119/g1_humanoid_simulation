# Bài toán ID-QP (Whole-Body Controller) cho G1

Tổng hợp cách bộ điều khiển trong `sim/idqp_core.py` (đứng cân bằng) và `sim/walk_core.py` /
`sim/stairs_core.py` (đi bộ, cầu thang) dựng và giải QP ở **mỗi bước mô phỏng 2 ms**.

---

## 1. Vị trí trong hệ thống

```
Tầng kế hoạch (offline, 1 lần lúc t = 0)          Tầng WBC (mỗi 2 ms)
───────────────────────────────────────          ─────────────────────────────────────────
bước chân ─► lịch pha ─► ZMP ref ─► preview  ─►   tra bảng theo t: c_ref, swing_ref, contact
                                    (LIPM)        ─► dựng QP từ trạng thái đo (q, q̇)
                                                  ─► OSQP ─► τ ─► d.ctrl = τ ─► mj_step
```

QP không biết gì về ZMP hay LIPM. Nó chỉ nhận **gia tốc mong muốn** của các task, cùng
**mô hình động lực học đầy đủ** của robot.

---

## 2. Biến quyết định

```
x = [ q̈ (35) | τ (29) | f (24) ]        NX = 88
```

| Khối | Kích thước | Ý nghĩa |
|---|---|---|
| `q̈` | nv = 35 | gia tốc tổng quát: 6 của floating base (pelvis) + 29 khớp |
| `τ` | nu = 29 | mô-men motor (actuator đã đổi sang motor thuần: `τ = ctrl`) |
| `f` | 3 × 8 = 24 | lực 3D tại **8 góc tiếp xúc** (4 quả cầu dưới mỗi bàn chân) |

Chỉ có `τ` được gửi xuống robot. `q̈` và `f` là biến phụ: chúng buộc `τ` phải nhất quán với
động lực học và với tiếp xúc.

Ma trận chọn khớp `S` (29×35) có `S[i, dof(i)] = 1`, nên `Sᵀτ` là lực tổng quát do motor sinh ra.

---

## 3. Dạng chuẩn OSQP

```
min_x  ½ xᵀ P x + gᵀ x
s.t.   lo ≤ A x ≤ hi
```

Ràng buộc đẳng thức được viết với `lo = hi`.

---

## 4. Hàm cost

### 4.1. Mỗi task là một bình phương tối thiểu

Mọi task đều có dạng "muốn gia tốc của một đại lượng `y` bằng `a_des`":

```
y = J q            ⇒   ÿ = J q̈ + J̇ q̇
cost_task = w · ‖ J q̈ + J̇ q̇ − a_des ‖²   =   w · ‖ A x − b ‖²
với  A = [ J | 0 | 0 ],   b = a_des − J̇ q̇
```

`a_des` luôn là **feedforward + PD**:

```
a_des = ÿ_ref + Kp (y_ref − y) + Kd (ẏ_ref − ẏ)
```

`J̇ q̇` được tính bằng **sai phân hữu hạn** (`dJqd`): dịch `q` một bước `ε·q̇` bằng
`mj_integratePos`, tính lại Jacobian, rồi lấy `(J₁ − J₀)/ε · q̇`.

### 4.2. Khai triển thành P, g (nguồn gốc của hệ số 2)

```
w‖Ax − b‖² = w (xᵀAᵀAx − 2bᵀAx + bᵀb)
```

So với `½xᵀPx + gᵀx`:

```
P += 2 w AᵀA
g −= 2 w Aᵀb          (bỏ hằng số bᵀb)
```

Code tương ứng:

```python
for w, A, b in tasks.values():
    P += 2 * w * A.T @ A
    g -= 2 * w * A.T @ b
```

### 4.3. Danh sách task

| Task | `y` / Jacobian | `a_des` | Kp / Kd | Trọng số `w` | Có ở |
|---|---|---|---|---|---|
| **CoM** | CoM toàn thân, `mj_jacSubtreeCom` (3×35) | `c̈_ref + Kp(c_ref − c) + Kd(ċ_ref − ċ)` | 100 / 20 | **100** | đứng, đi |
| **Hướng thân** (torso) | vận tốc góc torso, `jacBody` rot (3×35) | `Kp·e_rot − Kd·ω` (ref = thẳng đứng, yaw 0) | 200 / 30 | **20** | đứng, đi |
| **Hướng pelvis** | vận tốc góc pelvis (3×35) | như trên | 200 / 30 | **10** | đi |
| **Chân bay** | 6D của cổ chân bay (6×35) | tuyến tính: `p̈_ref + Kp(p_ref − p) + Kd(ṗ_ref − ṗ)`; góc: `Kp·e_rot − Kd·ω` | 400 / 40 | **100** | chỉ pha SS |
| **Tư thế** (posture) | 29 khớp, `A = diag(√w_j)` trên `q̈` khớp | `Kp(q_nom − q) − Kd q̇` | 60 / 15 | chân **0.05**, còn lại **1** | đứng, đi |

Ghi chú:
- Sai số hướng `e_rot` lấy từ `mju_subQuat(target, quat)`, đổi sang hệ world bằng `xmat`.
- Trọng số riêng cho từng khớp của task tư thế được đưa vào **bên trong** `A` và `b` (nhân với
  `√w_j`), còn trọng số ngoài bằng 1. Chân có trọng số rất nhỏ (0.05) để không cản các task CoM
  và chân bay.
- Task chân bay **chỉ được thêm khi đang ở pha một chân**.
- Bản cầu thang ghi đè `c_ref.z` bằng `com_z(t)`, cùng 3 đoạn quỹ đạo chân bay (nhấc / giữ / hạ).

### 4.4. Regularization

```
P[q̈, q̈] += 2 · 1e-6 · I
P[τ, τ]  += 2 · 1e-4 · I
P[f, f]  += 2 · 1e-5 · I
```

Tương đương cost `w_qdd‖q̈‖² + w_tau‖τ‖² + w_f‖f‖²`. Tác dụng:

1. **P xác định dương** ⇒ QP lồi chặt ⇒ **nghiệm duy nhất**. Nếu thiếu, sẽ có vô số cách chia lực
   giữa 8 góc chân cho cùng một tổng lực và mô-men.
2. Chọn nghiệm "rẻ" nhất: lực phân bố đều, mô-men nhỏ.

Các trọng số này rất nhỏ so với các task (100), nên gần như không làm lệch việc bám task.

### 4.5. Tổng hợp cost

```
J(x) = 100‖c̈ − a_com‖² + 20‖ω̇_torso − a_rot‖² + [10‖ω̇_pelvis − a_pv‖²] + [100‖ẍ_swing − a_sw‖²]
       + Σ_j w_j (q̈_j − a_post,j)²
       + 1e-6‖q̈‖² + 1e-4‖τ‖² + 1e-5‖f‖²
```

Đây là **WBC có trọng số** (weighted). Khi các task xung đột, QP thỏa hiệp theo `w`, không có ưu
tiên tuyệt đối.

---

## 5. Ràng buộc

`A` được xếp theo các nhóm. Tên nhóm (`dyn`, `feet`, `tau`, `fric`, `lim`) cũng là tên dùng trong
log và trong chẩn đoán infeasible.

### 5.1. Động lực học (`dyn`, 35 hàng, đẳng thức)

```
M(q) q̈ + h(q, q̇) = Sᵀ τ + Jcᵀ f
⇔  [ M | −Sᵀ | −Jcᵀ ] x = −h
```

- `M`: `mj_fullM`. `h = qfrc_bias` (Coriolis + trọng lực).
- `Jc` (24×35): Jacobian điểm của 8 góc chân (`mj_jac` tại vị trí góc trong world).
- **6 hàng đầu** (floating base, không có motor) chính là **cân bằng động lượng centroidal**:
  `m(c̈ + g) = Σf`, `L̇ = Σ(p_i − c) × f_i`. Đây là nơi CoM, lực chân và CoP bị trói buộc với nhau.
  Robot chỉ "đẩy" được thân bằng lực tiếp xúc.

### 5.2. Chân tiếp xúc không trượt (`feet`, 6 hàng mỗi chân, đẳng thức)

```
Jfoot q̈ + J̇foot q̇ = −kf · Jfoot q̇          (kf = 20)
⇔  [ Jfoot | 0 | 0 ] x = −J̇foot q̇ − kf Jfoot q̇
```

- Gia tốc 6D của bàn chân = 0. Số hạng Baumgarte `−kf·v` kéo vận tốc trôi về 0.
- Chỉ ép **vận tốc**, không ép vị trí. Nếu chân chưa chạm đất mà vẫn được khai báo tiếp xúc, QP sẽ
  coi như nó đứng yên trên không.
- Khi đi bộ, chỉ **chân trụ** có nhóm này. Đứng yên: 12 hàng. Pha SS: 6 hàng.

### 5.3. Giới hạn mô-men (`tau`, 29 hàng, bất đẳng thức hộp)

```
−τ_max ≤ τ ≤ τ_max          τ_max = actuator_forcerange · tau_scale
```

Ví dụ: gối 139 Nm, hip pitch 88 Nm, ankle 50 Nm.

### 5.4. Ma sát và lực đẩy một chiều (`fric`, 5 hàng mỗi góc × 8 = 40 hàng)

Nón ma sát được **tuyến tính hóa thành chóp 4 mặt**:

```
 f_x − μ f_z ≤ 0       −f_x − μ f_z ≤ 0
 f_y − μ f_z ≤ 0       −f_y − μ f_z ≤ 0
 0 ≤ f_z ≤ fz_max      (fz_max = 600 N)
```

- μ trong QP = 0.5 (bảo thủ), trong khi sàn MuJoCo có μ = 1.0.
- `f_z ≥ 0` tại các góc ⇒ **CoP tự động nằm trong đa giác đỡ**. Đây là ràng buộc ZMP duy nhất,
  và nó được ngầm định.
- **Chân bay**: cận trên của `f_z` ở 4 góc của nó đặt bằng 0 ⇒ `f = 0`. Ma trận không đổi, chỉ đổi
  `hi`.

### 5.5. Giới hạn khớp qua gia tốc (`lim`, 29 hàng, bất đẳng thức hộp trên `q̈` khớp)

Nhìn trước một khoảng `h_lim = 0.08 s` với gia tốc không đổi:

```
q + q̇ h + ½ q̈ h² ∈ [q_lo, q_hi]
⇒  2(q_lo − q − q̇h)/h² ≤ q̈ ≤ 2(q_hi − q − q̇h)/h²
```

- `q_lo`, `q_hi` = biên khớp trong XML, thu vào thêm 0.05 rad. Gối có biên dưới 0.25 rad để không
  duỗi thẳng (tránh kỳ dị).
- Nếu cận dưới vượt cận trên (khi đã ở ngoài biên), code kẹp `lo = min(lo, hi)` để không tự gây
  infeasible.

### 5.6. Kích thước

| Tình huống | dyn | feet | tau | fric | lim | **Tổng hàng** |
|---|---|---|---|---|---|---|
| Đứng / DS | 35 | 12 | 29 | 40 | 29 | **145** |
| Pha SS (1 chân) | 35 | 6 | 29 | 40 | 29 | **139** |

Biến: 88. Ràng buộc đẳng thức: 47 (đứng) hoặc 41 (SS).

---

## 6. Giải và xử lý kết quả (`solve_qp`)

```python
solver = osqp.OSQP()
solver.setup(csc(triu(P)), g, csc(A), lo, hi,
             eps_abs=1e-6, eps_rel=1e-6, polishing=True, max_iter=20000)
res = solver.solve()
```

- Mỗi bước **dựng lại từ đầu**: không warm start, không cập nhật ma trận. Mất khoảng 2–5 ms, nên
  chậm hơn thời gian thực.
- Phân loại trạng thái:

| `kind` | Status OSQP | Xử lý |
|---|---|---|
| `ok` | solved | dùng `τ = x[τ]` |
| `warn` | solved inaccurate / max iter | vẫn dùng `x` nếu hữu hạn |
| `infeasible` | primal infeasible (inaccurate) | **giữ `τ` của bước trước** (fallback); dùng chứng chỉ Farkas `prim_inf_cert` để chỉ ra nhóm ràng buộc gây mâu thuẫn |
| `error` | còn lại | dừng mô phỏng |

- Thông tin trả về cho monitor: `qdd`, `tau_qp`, `f` (8×3), `cdd_qp`, phần dư từng task,
  số ràng buộc đang active theo nhóm, số vòng lặp, thời gian giải.

---

## 7. Khác biệt giữa đứng và đi bộ

| | Đứng (`IDQPController`) | Đi bộ (`WalkController`) |
|---|---|---|
| `c_ref` | cố định / theo lịch đẩy | bảng preview LIPM (x, y), z = `ZC` |
| Task | CoM, torso, posture | + pelvis, + chân bay (pha SS) |
| `feet` | 2 chân luôn tiếp xúc | theo `contact_plan(t)` |
| `fric` hi | `fz_max` ở mọi góc | 0 ở 4 góc của chân bay |
| Nhóm hàng | cố định | `groups`, `row_names` dựng lại theo pha |

---

## 8. Hạn chế đã biết

1. **Không có task động lượng góc** `L`. Thân trên và tay không chủ động giúp giữ thăng bằng.
2. **Chỉ ưu tiên theo trọng số**. CoM có thể bị hy sinh cho chân bay (thấy rõ ở cầu thang).
3. **Khâu PD của CoM khuếch đại sai số thành sai lệch ZMP**:
   `p_thực ≈ p_ref − (z_c/g)(Kp·e + Kd·ė)`. Với `z_c/g ≈ 0.068` và Kp = 100, sai số CoM 1 cm làm
   ZMP lệch 6.8 cm, trong khi bàn chân chỉ rộng 6 cm.
4. **Tiếp xúc theo lịch**, không theo cảm biến. Ràng buộc `feet` chỉ ép vận tốc.
5. **Dựng lại OSQP mỗi bước**, nên chưa đạt thời gian thực.

---

## 9. Bảng tra nhanh vị trí trong code

| Nội dung | File |
|---|---|
| Config: gain, trọng số, μ, `h_lim` | `sim/idqp_core.py` — `Config` |
| Ma trận `S`, actuator → motor, 8 góc chân | `sim/idqp_core.py` — `IDQPController.__init__` |
| `A_fric`, `A_tau`, `A_lim`, biên khớp | `sim/idqp_core.py` — `_build_constant_constraints` |
| Jacobian, `J̇q̇`, Jacobian góc chân | `task_jacobians`, `dJqd`, `corner_jac` |
| Dựng cost và ràng buộc khi đứng | `sim/idqp_core.py` — `solve` |
| Gọi OSQP, fallback, chẩn đoán | `sim/idqp_core.py` — `solve_qp`, `diagnose` |
| Dựng cost và ràng buộc khi đi (theo pha) | `sim/walk_core.py` — `solve` |
| Quỹ đạo chân bay / CoM z cầu thang | `sim/stairs_core.py` — `swing_pos`, `com_z` |
