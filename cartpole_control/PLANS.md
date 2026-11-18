# Kế hoạch cart-pole (chặng 2, bài 1)

Kế hoạch có 7 bước. Bạn tự viết code cho mọi bước, còn ở đây chỉ ghi việc cần làm và tiêu chí để biết đã xong. Mỗi bước chạy được và đạt tiêu chí thì mới sang bước sau. Tổng cộng khoảng 5–6 buổi (2–3 giờ mỗi buổi). Phần còn lại của tuần 3–4 dành cho acrobot.

**Đọc trước:** Underactuated Robotics ch. 3 (Acrobots, Cart-Poles: energy shaping, partial feedback linearization) và ch. 8 (LQR). Nếu muốn đọc trước phần vùng hút thì xem ch. 9.

**Cấu trúc thư mục:**
```
cartpole_control/
  model/cartpole.xml     # model MJCF (bước 1, đã xong)
  params.py              # thông số vật lý, khớp với cartpole.xml
  dynamics.py            # M, C, g, f(x, u), năng lượng, viết từ Lagrange
  integrators.py         # Euler, RK4 tự viết (bước 3)
  linearize.py           # A, B tính tay và bằng sai phân
  controllers.py         # LQR, energy swing-up, logic chuyển chế độ
  scripts/view.py        # xem model, ctrl = 0
  sim_utils.py           # helper: load model, chạy vòng kín, lưu đồ thị
  scripts/stepN_*.py     # mỗi bước một script chạy, kiểm tra và vẽ đồ thị
  scripts/demo.py        # xem bộ điều khiển trực tiếp trong viewer
  outputs/               # đồ thị, video, K/S/rho của bước 5 (tạo khi chạy)
  notes/cartpole.md      # ghi chú và câu trả lời
```
Chạy mọi script từ thư mục `cartpole_control/` với venv chung: `source ../.venv/bin/activate`.

**Tiến độ** (tick khi đạt tiêu chí kiểm tra):
- [x] Bước 1: model MJCF
- [x] Bước 2: phương trình chuyển động
- [x] Bước 3: mô phỏng thụ động và năng lượng
- [x] Bước 4: tuyến tính hóa tại đỉnh
- [x] Bước 5: LQR giữ đỉnh
- [x] Bước 6: swing-up bằng năng lượng
- [x] Bước 7: ghép swing-up với LQR

---

### Bước 1: Tự viết model MJCF (½ buổi) ✅

Đã xong: [model/cartpole.xml](model/cartpole.xml), thông số trong [README.md](README.md). `nq = 2, nv = 2, nu = 1`; thả từ `up_tilt` thì con lắc đổ và dao động qua đáy.

- Gồm một khớp `slide` cho xe, một khớp `hinge` cho con lắc và một actuator `motor` tác động lên xe. Đặt `ctrlrange` khoảng ±20 N và giới hạn ray, ví dụ ±2 m.
- Khai báo `<inertial>` tường minh (khối lượng, tâm khối, quán tính). Nếu để MuJoCo tự tính từ geom thì bước 2 sẽ không khớp được tới 1e-8.
- Đặt damping bằng 0, armature bằng 0, `timestep` 0.001–0.002.
- **Kiểm tra:** in `nq, nv, nu`. Thả con lắc lệch nhẹ khỏi đỉnh với `ctrl = 0`, nó phải đổ xuống và dao động quanh đáy.

### Bước 2: Phương trình chuyển động (1 buổi)
- Dùng Lagrange để rút ra `M(q)q̈ + C(q,q̇)q̇ + g(q) = Bu`, với `q = [x, θ]`.
- Chọn quy ước góc ngay từ đầu và ghi vào notes. Tedrake dùng `θ = 0` ở đáy và `θ = π` ở đỉnh. Cần đối chiếu chiều dương của `hinge` trong MuJoCo với chiều trong phép rút gọn của bạn.
- Viết `M, C, g` bằng NumPy và so với `mj_fullM` và `qfrc_bias` tại 100 trạng thái ngẫu nhiên.
- **Kiểm tra:** sai số nhỏ hơn 1e-8 (đúng tiêu chí của chặng 1).

### Bước 3: Mô phỏng thụ động và năng lượng (½ buổi)
- Tự tích phân `f(x, 0)` bằng Euler, rồi bằng RK4. Tính năng lượng `E = T + V` theo thời gian.
- **Kiểm tra:** chạy 10 s với RK4, năng lượng trôi dưới 1e-3. Euler phải trôi rõ rệt, và bạn giải thích được vì sao.

### Bước 4: Tuyến tính hóa tại đỉnh (½ buổi)
- Tính `A, B` bằng tay, rồi so với sai phân hữu hạn. Có thể tự viết sai phân, hoặc lấy bản rời rạc bằng sai phân quanh `mj_step` (`linearize_mujoco_step`). `mujoco.mjd_transitionFD` không hỗ trợ bộ tích phân RK4.
- **Kiểm tra:**
  - `A, B` tính tay khớp với sai phân.
  - Có đúng một trị riêng thực dương (hệ không ổn định).
  - Ma trận điều khiển được `[B, AB, A²B, A³B]` có hạng 4.

### Bước 5: LQR giữ đỉnh (1 buổi)
- Viết LQR liên tục với `solve_continuous_are`. Viết thêm bản rời rạc dùng `solve_discrete_are`, với `A, B` rời rạc lấy từ `linearize_mujoco_step`.
- Trạng thái lỗi là `x̃ = [x, θ − π, ẋ, θ̇]`. **Phải quấn `θ − π` về khoảng [−π, π]**, nếu không robot sẽ "quay vòng" khi góc vượt quá ±π.
- Thí nghiệm:
  - Đổi `Q, R` và quan sát đánh đổi giữa tốc độ hội tụ và độ lớn lực.
  - Quét lưới góc ban đầu để tìm vùng hút thật, rồi vẽ lại.
  - Đẩy xe bằng `xfrc_applied`.
- **Kiểm tra:** bắt đầu từ `θ = π ± 0.2` rad, hệ ổn định trong 3 s và lực không chạm `ctrlrange`. Vẽ được vùng hút trên mặt phẳng `(θ, θ̇)`.

### Bước 6: Swing-up bằng năng lượng (1 buổi)
- Rút ra **partial feedback linearization (collocated)**: tìm lực `u` sao cho `ẍ = ẍ_des`. Bước này là tiền thân của computed torque và ID-QP sau này.
- Luật energy shaping: `ẍ_des = k·Ẽ·θ̇·cos θ`, với `Ẽ = E − E_đỉnh`. Cộng thêm một thành phần PD trên `x` để xe không chạy khỏi ray.
- **Kiểm tra:** bắt đầu từ đáy với `θ = 0.05`, vẽ `Ẽ(t)` hội tụ về 0 và xe nằm trong ±2 m.

### Bước 7: Ghép swing-up với LQR (1 buổi)
- Dùng swing-up cho tới khi `x̃ᵀ S x̃ < ρ`, rồi chuyển sang LQR. `S` là nghiệm Riccati, còn `ρ` chọn từ vùng hút đã đo ở bước 5.
- **Kiểm tra:**
  - Từ đáy, con lắc lên đỉnh và giữ được 10 s.
  - Không nhảy qua lại giữa hai chế độ ở ngưỡng chuyển.
  - Xe nằm trong ray, lực nằm trong `ctrlrange`.
  - Quay được video bằng `mediapy`.

---

### Ghi chú `notes/cartpole.md` cần trả lời được
1. Vì sao cart-pole là hệ thiếu actuator (2 bậc tự do, 1 actuator), và điều này liên quan gì tới 6 bậc tự do floating base của G1?
2. PFL collocated cho phép điều khiển trực tiếp biến nào? Vì sao không làm như vậy được với biến còn lại? (Liên hệ: "computed torque không dùng trực tiếp được cho floating base".)
3. `Q, R` ảnh hưởng thế nào tới các cực vòng kín? (Chuẩn bị cho ZMP preview ở chặng 5, vốn chính là một bài LQR.)
4. Vùng hút đo được lớn hay nhỏ hơn bạn đoán, và vì sao? (Gợi ý: giới hạn lực, phi tuyến.)

### Lỗi hay gặp
- Quy ước dấu hoặc gốc của `θ` lệch với MuJoCo, làm LQR đẩy sai chiều.
- Quên quấn góc.
- Dùng `A, B` liên tục cho LQR rời rạc, hoặc ngược lại.
- MuJoCo kẹp `ctrl` theo `ctrlrange`, nên lực mô phỏng khác lực bạn tính.
- Gọi `mj_step` xong mới đọc `qacc` cho phép so sánh. Muốn so tại đúng trạng thái đã đặt thì phải gọi `mj_forward`.
