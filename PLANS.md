# Lộ trình học và code điều khiển robot đi bộ (G1)

Nov 8, 2026 · @dat

## Mục tiêu

Sau khoảng 12 tuần (2–3 giờ mỗi ngày, cùng nhịp với kế hoạch 2 tuần hiện tại), bạn tự viết lại từ trang trắng được hai demo: G1 cân bằng bằng ID-QP và G1 đi bộ bằng LIPM + ZMP preview + ID-QP. Bạn hiểu từng dòng code và debug được khi robot ngã.

Cách dùng kế hoạch:

- Mỗi chặng có ba phần: **học** (chương sách, bài giảng), **code** (một bài tự viết, nhỏ, chạy được) và **kiểm tra** (một tiêu chí đo được). Chỉ qua chặng khi đạt tiêu chí.
- Đừng mở code demo có sẵn trước khi tự viết xong bài của chặng. Viết xong rồi mới so sánh.
- Kế hoạch 2 tuần đang làm (cart-pole, acrobot) chính là chặng 2. Nếu đã xong thì bỏ qua chặng này.
- Mỗi chặng ghi một trang ghi chú trong `notes/`, giống các ghi chú về quaternion và position actuator đã có.

## Bản đồ kiến thức

![[Pasted image 20261108212121.png]]

Hai nhánh độc lập gặp nhau ở chặng 6: ID-QP (động lực học + tối ưu) ở bên trái, mô hình đơn giản (LQR → ZMP preview) ở bên phải. Chặng 3 có thể học song song với chặng 2 nếu muốn rút ngắn lịch.

## Chặng 0: Công cụ (tuần 1, song song với chặng 1)

Mục tiêu là dùng thành thạo đúng bốn thư viện mà demo cần.

**Học**

- NumPy: ma trận, slicing, `np.linalg.solve`, lý do không dùng `inv`.
- MuJoCo Python: `MjModel` / `MjData`, `mj_step`, `mj_forward`, `qpos` (7 + n) và `qvel` (6 + n), `mj_fullM`, `qfrc_bias`, `mj_jac*`, `subtree_com`, đọc contact bằng `mj_contactForce`, `Renderer`.
- Pinocchio (tùy chọn): cùng các hàm `crba`, `rnea`, `computeJointJacobians`, dùng để kiểm tra chéo kết quả MuJoCo.
- OSQP: dạng chuẩn `min ½ xᵀPx + qᵀx, l ≤ Ax ≤ u`, ma trận thưa `scipy.sparse`.

**Code**

1. Nạp G1 từ Menagerie, in `nq, nv, nu`, tên các khớp và actuator.
2. Thả rơi robot với `ctrl = 0`, quay video 3 giây.
3. Giải một QP 2 biến bằng OSQP và so với nghiệm tính tay.

**Kiểm tra:** `M` từ `mj_fullM` đối xứng, xác định dương, và `M[0,0]` bằng tổng khối lượng robot (33.3 kg).

## Chặng 1: Động lực học robot (tuần 1–2)

Mục tiêu là hiểu và tự tính được mọi đại lượng trong phương trình `M(q)q̈ + C(q,q̇)q̇ + g(q) = Sᵀτ + Jᵀλ`.

**Học**

- Phương trình manipulator, Lagrange cho hệ nhiều khâu: UR ch. 1–3, phụ lục B. Modern Robotics (Lynch & Park) ch. 3–5 và 8.
- Xoay trong 3D: ma trận xoay, quaternion, vận tốc góc, vì sao `nq ≠ nv` (đã có ghi chú MuJoCo Quaternion).
- Jacobian hình học: ánh xạ `q̇ →` vận tốc điểm, `Jᵀf →` lực suy rộng, và `J̇q̇`.
- Floating base: 6 bậc tự do không có actuator, ma trận chọn `S`, CoM và Jacobian CoM.
- Tiếp xúc: ma sát Coulomb, nón ma sát, lực pháp tuyến đơn phương (chỉ đẩy, không kéo).

**Code**

1. Con lắc kép: tự viết `M, C, g` bằng tay, so với MuJoCo tại 100 trạng thái ngẫu nhiên.
2. G1: tính `J` của bàn chân và kiểm tra `J q̇` bằng sai phân vị trí.
3. Tính `J̇q̇` bằng sai phân theo hướng `q̇` (cách demo đang dùng) và so với `mj_jacDot`.
4. Robot đứng yên: giải bình phương tối thiểu `Sᵀτ + Jᵀλ = g` để ra torque giữ tư thế.

**Kiểm tra:** sai số `M, C, g` tự viết so với MuJoCo dưới 1e-8. Giải thích được vì sao 6 hàng đầu của phương trình không chứa `τ`.

## Chặng 2: Điều khiển cơ bản (tuần 3–4, trùng kế hoạch 2 tuần hiện tại)

Mục tiêu là nắm ba công cụ mà demo dùng liên tục: PD, computed torque và LQR.

**Học**

- PD và hệ bậc hai: `ω_n`, `ζ`, cách chọn `Kp`, `Kd`. Đây là lý do demo dùng `Kp = 100, Kd = 20` cho CoM (`ω = 10`, `ζ = 1`).
- Computed torque và partial feedback linearization: UR ch. 3. ID-QP chính là computed torque có thêm ràng buộc.
- LQR rời rạc và liên tục, phương trình Riccati, hàm `solve_discrete_are`: UR ch. 8.
- Vùng hút và Lyapunov cơ bản: UR ch. 9.

**Code**

1. Cart-pole và acrobot theo kế hoạch 2 tuần: LQR giữ đỉnh, swing-up năng lượng.
2. Tay máy 7 bậc tự do (Panda trong Menagerie): computed torque bám quỹ đạo khớp, so với PD thuần.
3. G1 treo cố định (fixed base): PD cộng bù trọng lực `τ = Kp(q* − q) − Kd q̇ + g(q)` cho tất cả 29 khớp.

**Kiểm tra:** computed torque bám quỹ đạo với sai số nhỏ hơn PD thuần ít nhất 10 lần. Nói được vì sao computed torque không dùng được trực tiếp cho robot floating base.

## Chặng 3: Tối ưu lồi và QP (tuần 5)

Mục tiêu là viết được một bài toán điều khiển thành QP chuẩn và hiểu solver làm gì bên trong.

**Học**

- Bình phương tối thiểu có trọng số: `∑ wᵢ‖Aᵢx − bᵢ‖²` biến thành `P = 2∑wᵢAᵢᵀAᵢ`, `q = −2∑wᵢAᵢᵀbᵢ`. Đây là đoạn code xây cost trong demo.
- Tập lồi, hàm lồi, điều kiện KKT, nhân tử Lagrange: Boyd & Vandenberghe, _Convex Optimization_, ch. 1–5 (đọc lướt ch. 2–3, kỹ ch. 4–5).
- Xấp xỉ nón ma sát bằng kim tự tháp để biến ràng buộc nón thành tuyến tính. Biết thêm SOCP nếu muốn giữ nón tròn.
- Thuật toán: active-set (qpOASES), interior point, ADMM (OSQP). Hiểu vì sao OSQP nhanh và vì sao cần `polishing`.
- Ưu tiên trọng số so với ưu tiên cứng (hierarchical QP).

**Code**

1. Tự viết một QP chỉ có ràng buộc đẳng thức bằng cách giải hệ KKT bằng NumPy, so với OSQP.
2. Phân bổ lực tĩnh: robot đứng trên 8 góc bàn chân. Tìm 8 lực tiếp xúc sao cho tổng lực và mômen cân bằng trọng lực, nằm trong nón ma sát, tổng bình phương nhỏ nhất.
3. Dịch CoM dần ra mép bàn chân và xem QP báo infeasible đúng lúc CoM ra ngoài đa giác đỡ.

**Kiểm tra:** nghiệm KKT tự viết trùng OSQP tới 1e-6. Bài 3 cho ra biên đa giác đỡ đúng với hình học bàn chân.

## Chặng 4: Whole-body control bằng ID-QP (tuần 6–7)

Mục tiêu là tự viết lại demo G1 cân bằng: đứng yên, dịch CoM, ngồi xuống, chịu được cú đẩy 60 N.

**Học**

- Inverse dynamics QP: biến `[q̈, τ, λ]`, ràng buộc động lực học, tiếp xúc, ma sát, giới hạn.
- Task space control: mỗi task là `J q̈ + J̇ q̇ = a_des`, với `a_des` từ PD cộng feedforward.
- Đọc: Feng, Atkeson et al. 2014 và Kuindersma, Tedrake et al. 2016 về bộ điều khiển Atlas của MIT. Tài liệu và ví dụ của TSID (Pinocchio).
- Baumgarte stabilization cho ràng buộc tiếp xúc.
- Động lượng centroidal `h_G = A_G q̇` (Orin & Goswami 2013), để chuẩn bị cho task động lượng góc.

**Code** (làm theo thứ tự, mỗi bước chạy được mới sang bước sau)

1. Đổi actuator G1 sang motor. Chỉ có task tư thế và hai chân đứng yên.
2. Thêm task CoM, rồi task hướng thân.
3. Thêm giới hạn torque và giới hạn khớp dạng gia tốc.
4. Thử thách: dịch CoM 5 cm, ngồi 6 cm, đẩy 60 N.
5. Tự tạo lại hai lỗi đã gặp trong demo: task tư thế đánh nhau với task CoM, và gối kẹt ở giới hạn khớp khi chưa có ràng buộc khớp. Giải thích được nguyên nhân.
6. Mở rộng: task động lượng góc `‖k̇_G‖²` và xem tay tự vung khi bị đẩy.

**Kiểm tra:** lỗi bám CoM dưới 5 mm khi dịch và ngồi. Lực pháp tuyến QP tính ra khớp với lực MuJoCo đo được. Robot hồi phục sau cú đẩy 60 N.

## Chặng 5: Mô hình đơn giản cho đi bộ (tuần 8–9)

Mục tiêu là hiểu tại sao robot ngã, và sinh được quỹ đạo CoM cân bằng cho một lịch bước chân cho trước.

**Học**

- Đi bộ thụ động: rimless wheel, compass gait, Poincaré map, va chạm: UR ch. 4. Đây là bước 3 trong kế hoạch 2 tuần.
- LIPM, ZMP, đa giác đỡ, cart-table: UR ch. 5. Kajita et al. 2003, _Biped walking pattern generation by using preview control of ZMP_.
- Preview control: LQR có tích phân lỗi và nhìn trước ZMP tương lai. Katayama 1985 cho đạo hàm độ lợi preview.
- Capture point và DCM: Pratt 2006, Englsberger 2015. Đây là nền cho việc điều chỉnh bước chân khi bị đẩy.

**Code** (tất cả trên mô hình điểm, chưa dùng robot)

1. Compass gait đi xuống dốc, tìm chu kỳ giới hạn bằng Poincaré map.
2. LIPM 2D: mô phỏng CoM khi ZMP cố định, thấy nghiệm phân kỳ theo `e^(ωt)` với `ω = √(g/z_c)`.
3. ZMP preview control 1D, rồi 2D: lịch bước chân → ZMP tham chiếu → quỹ đạo CoM. Vẽ giống hình trong bài Kajita.
4. Capture point: cho LIPM bị đẩy, tính chỗ đặt chân để dừng lại trong một bước.

**Kiểm tra:** ZMP sinh ra từ quỹ đạo CoM bám ZMP tham chiếu với sai số dưới 5 mm. Giải thích được vì sao CoM phải bắt đầu dịch trước khi ZMP dịch (non-minimum phase).

## Chặng 6: Ghép thành demo đi bộ (tuần 10–11)

Mục tiêu là nối quỹ đạo CoM của chặng 5 vào ID-QP của chặng 4, để G1 đi 10 bước, mỗi bước 15 cm.

**Học**

- Máy trạng thái pha: đứng hai chân và đứng một chân, thời điểm nhấc và chạm chân.
- Quỹ đạo chân bay: spline bậc 5, vận tốc và gia tốc bằng 0 ở hai đầu.
- Đổi tiếp xúc trong QP: bỏ ràng buộc đứng yên và ép lực bằng 0 cho chân bay, thêm task bám quỹ đạo chân bay.
- Đo ZMP từ lực tiếp xúc để kiểm chứng mô hình.

**Code** (theo đúng thứ tự debug của demo)

1. Dồn trọng tâm qua lại tại chỗ, hai chân vẫn chạm đất.
2. Đứng một chân 2 giây.
3. Bước tại chỗ (bước dài 0).
4. Bước 5 cm, rồi 10 cm, rồi 15 cm.
5. Vẽ ZMP tham chiếu so với ZMP đo, quỹ đạo chân và lực pháp tuyến.
6. Tự tạo lại lỗi chân bay nhấc quá chậm làm chân trụ rung lực, rồi sửa.

**Kiểm tra:** đi 10 bước không ngã, ZMP đo luôn nằm trong bàn chân trụ, lỗi CoM dưới 5 mm. Chịu được cú đẩy ngang 25 N khi đang đi.

## Chặng 7: Nâng cao (tuần 12 trở đi, chọn theo hứng thú)

Sau chặng 6 bạn đã có đủ nền để đọc và code các hướng hiện đại. Mỗi dòng dưới đây là một dự án 1–3 tuần, xếp từ gần với code hiện tại đến xa nhất.

|   |   |   |
|---|---|---|
|Dự án|Học|Kết quả cần đạt|
|Điều chỉnh bước chân bằng DCM|Englsberger 2015, UR ch. 5|G1 chịu được cú đẩy ngang 40 N khi đang đi|
|MPC trên LIPM, vị trí chân là biến|Wieber 2006, Herdt 2010|Thay preview control, ràng buộc ZMP cứng, tự chọn chỗ đặt chân|
|Convex MPC trên mô hình một khối cứng|Di Carlo 2018 (MIT Cheetah)|Đi nhanh hơn 0.5 m/s, đi quay|
|Tối ưu quỹ đạo centroidal|UR ch. 10, Dai & Tedrake 2014|CoM lên xuống tự nhiên, bước lên bậc|
|MPC toàn thân bằng DDP|Crocoddyl, Mastalli 2020|Bỏ mô hình đơn giản, tối ưu trên động lực học đầy đủ|
|Contact-implicit trajectory optimization|UR, Posa & Tedrake 2014|Bộ tối ưu tự tìm dáng đi, không cho trước lịch bước|
|Lập kế hoạch bước chân bằng mixed-integer|UR, Deits & Tedrake 2014|Đi qua địa hình có vùng cấm đặt chân|
|Học tăng cường|mjlab / MuJoCo Playground, `g1_mjlab.xml`|Policy đi bộ RL, so sánh với bộ điều khiển dựa trên mô hình|

Thứ tự gợi ý là DCM, rồi MPC trên LIPM, rồi DDP. Robot thật (ước lượng trạng thái, sim-to-real) là bước 6 trong lộ trình tổng thể, nằm ngoài kế hoạch này.

### Dự án MPC trên LIPM: hiện trạng và việc tiếp theo (cập nhật 2026-11-18)

**Đã có:** `sim/mpc_core.py` và `sim/run_g1_mpc.py`.

- Cost: α‖U‖² + γ‖Z − Z_ref‖², α/γ = 1e-6, biến là jerk.
- Ràng buộc: ZMP nằm trong đa giác đỡ (lề 5 mm), và |ξ_N − ξ_ref| ≤ 10 mm ở cuối chân trời.
- T = 0.05 s, N = 32. Với T = 0.1 s, ZMP ra ngoài đa giác 15 mm vì pha DS chỉ dài 0.2 s.
- Lịch bước cố định.
- Đi bình thường: 8/8 bước PASS cả vòng hở lẫn vòng kín. MPC giải mất khoảng 2.8 ms (Python).
- Bỏ ràng buộc cuối chân trời thì ξ_N dự đoán lệch tới 47 cm. Robot vẫn đi được, nhưng ràng buộc này là cần.

**Vấn đề:** khi bị đẩy, bản vòng kín (x̂₀ lấy từ CoM đo được) kém hơn preview.

- J_safe ở hướng −y theo thứ tự DS / đầu SS / giữa SS / cuối SS: MPC đạt 4/4/5/6 N·s, preview đạt ≥8/5/6/≥8 N·s.
- Cùng MPC đó ở vòng hở thì ngang preview.
- Nguyên nhân: mỗi lần giải lại, c_ref bị đặt bằng CoM đo được, nên bộ PD CoM trong ID-QP mất tác dụng. MPC chỉ được hồi phục với ZMP trong bàn chân, trong khi robot thật còn dùng được động lượng góc (xoay hông).
- Lề ZMP, ε và ràng buộc cuối chân trời gần như không ảnh hưởng.
- Số liệu: `out/resilience_mpc.csv` và bảng biến thể trong lịch sử phiên làm việc.

**Việc tiếp theo** (theo thứ tự):

1. **Khép vòng một phần:** x̂₀ = x_plan + β(x_đo − x_plan), quét β ∈ {0, 0.2, 0.5}. Chạy lại `resilience_g1.py --controllers mpc --csv resilience_mpc_beta.csv` cho từng β. Đạt yêu cầu khi β tốt nhất không kém preview ở ô nào.
2. **MPC vòng hở + DCM feedback ở tầng dưới:** MPC chỉ sinh tham chiếu (c, ξ_ref), lớp DCM xử lý sai số bằng r_cmd = r_ref + (1 + k/ω)e như `dcm_core.py`. So bảng Δv với hướng 1.
3. **Dời bước (Herdt 2010):** thêm vị trí bước F làm biến, cost δ‖F − F_nom‖², ràng buộc `step_dx` / `step_width`. Đây là cách duy nhất cải thiện được ô +y đầu SS, hiện mọi biến thể đều kẹt ở 3 N·s, còn `dcm_step` đạt 4. Mục tiêu: vượt `dcm_step` trên toàn bảng.
4. (Tuỳ chọn) Tăng tốc MPC: giữ nguyên cấu trúc ma trận A và dùng `solver.update()` thay cho setup lại mỗi lần, vector hoá việc dựng ràng buộc.
5. (Tuỳ chọn) Mở rộng đo resilience: thêm đẩy dọc ±x, đẩy ở nhiều bước, nâng lưới J lên 12 vì các ô đạt ≥ 8 mới chỉ là chặn dưới.

## Tiến độ và tài liệu

Tick mỗi chặng khi đã đạt tiêu chí kiểm tra của nó.

- Chặng 0–1: công cụ và động lực học robot (tuần 1–2)
- Chặng 2: điều khiển cơ bản, cart-pole và acrobot (tuần 3–4)
- Chặng 3: tối ưu lồi và QP (tuần 5)
- Chặng 4: ID-QP, G1 cân bằng (tuần 6–7)
- Chặng 5: LIPM, ZMP preview, capture point (tuần 8–9)
- Chặng 6: G1 đi bộ (tuần 10–11)
- Chặng 7: dự án nâng cao đầu tiên (tuần 12+)

Tài liệu chính (liệt kê theo trí nhớ, chưa mở lại từng trang trong lần viết này):

- Russ Tedrake, [Underactuated Robotics](https://underactuated.csail.mit.edu), MIT 6.832
- Lynch & Park, [Modern Robotics](http://hades.mech.northwestern.edu/index.php/Modern_Robotics)
- Boyd & Vandenberghe, [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/)
- [MuJoCo documentation](https://mujoco.readthedocs.io) và [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- [OSQP](https://osqp.org), [Pinocchio](https://github.com/stack-of-tasks/pinocchio), [TSID](https://github.com/stack-of-tasks/tsid), [Crocoddyl](https://github.com/loco-3d/crocoddyl)
- Kajita et al. 2003 (ZMP preview), Pratt et al. 2006 (capture point), Englsberger et al. 2015 (DCM), Kuindersma et al. 2016 (Atlas), Orin & Goswami 2013 (centroidal)
- Code demo của dự án này: `g1_idqp_demo/g1_idqp_balance.py` và `g1_idqp_demo/walk/g1_idqp_walk.py` trong thư mục dự án