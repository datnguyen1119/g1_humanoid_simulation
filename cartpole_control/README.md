# Cart-pole control

Chặng 2 trong lộ trình [`../PLANS.md`](../PLANS.md): LQR giữ đỉnh, swing-up bằng năng lượng, rồi ghép hai bộ điều khiển. Làm trước phần humanoid.

## Chạy

```bash
cd ~/Learning/humanoid_g1_control/cartpole_control
source ../.venv/bin/activate       # venv chung của dự án, xem ../requirements.txt
python scripts/view.py             # bắt đầu gần đỉnh (lệch 0.1 rad), ctrl = 0, con lắc đổ xuống
python scripts/view.py --key down  # treo yên ở đáy
python scripts/view.py --key up    # đúng đỉnh (cân bằng không ổn định)
```

## Mô hình: [model/cartpole.xml](model/cartpole.xml)

| Ký hiệu | Giá trị | Ghi chú |
|---|---|---|
| `m_c` | 1.0 kg | khối lượng xe |
| `m_p` | 0.2 kg | khối lượng con lắc (thanh đồng chất) |
| `L` | 0.6 m | chiều dài con lắc |
| `l` | 0.3 m | khoảng cách từ khớp tới tâm khối con lắc |
| `I` | 0.006 kg·m² | quán tính con lắc quanh tâm khối, `m_p L²/12` |
| `g` | 9.81 m/s² | |
| `u` | ±20 N | lực ngang lên xe (actuator `force`) |
| ray | ±2 m | giới hạn khớp `slider` |
| `dt` | 0.002 s | bộ tích phân RK4 |

- `q = [x, θ]`, `qvel = [ẋ, θ̇]`.
- `θ = 0` là treo xuống, `θ = π` là thẳng đứng lên. Đầu con lắc ở `(x + L sin θ, −L cos θ)`; trục khớp `hinge` là `−y` để có quy ước này.
- Không có ma sát, damping hay armature, và tắt va chạm. Quán tính khai báo tường minh bằng `<inertial>`, geom chỉ để hiển thị. Nhờ vậy `M, C, g` tự rút ra phải khớp `mj_fullM` và `qfrc_bias` tới 1e-8.
- Keyframe: `down`, `up`, `up_tilt` (`θ = π − 0.1`).

## Chạy lời giải

```bash
cd ~/Learning/humanoid_g1_control/cartpole_control
source ../.venv/bin/activate
python scripts/step2_check_dynamics.py   # M, C, g so với MuJoCo
python scripts/step3_energy.py           # Euler và RK4 tự viết, năng lượng
python scripts/step4_linearize.py        # A, B, trị riêng, tính điều khiển được
python scripts/step5_lqr.py              # LQR, quét Q/R, đẩy xe, vùng hút (~25 s)
python scripts/step6_swingup.py          # swing-up bằng năng lượng
python scripts/step7_full.py             # swing-up rồi LQR, quay video
python scripts/demo.py --mode full       # xem trực tiếp (full | lqr | swingup)
```
Mỗi script in `[PASS]` / `[FAIL]` theo tiêu chí trong [PLANS.md](PLANS.md), đồ thị và video lưu vào `outputs/`.

## Thư mục

```
model/cartpole.xml   model MJCF
params.py            thông số, khớp với cartpole.xml
dynamics.py          M, C, g, f(x, u), năng lượng
integrators.py       Euler, RK4 tự viết
linearize.py         A, B tại đỉnh (tính tay, sai phân, sai phân quanh mj_step)
controllers.py       LQR, PFL, swing-up, chuyển chế độ; Q, R và hệ số đã chỉnh
sim_utils.py         load model, chạy vòng kín, lưu đồ thị
scripts/view.py      xem model, ctrl = 0
scripts/demo.py      xem bộ điều khiển trực tiếp
scripts/stepN_*.py   script chạy và kiểm tra từng bước
notes/cartpole.md    ghi chú, kết quả, câu hỏi cần trả lời
outputs/             đồ thị, video (tạo khi chạy)
```
