# Cart-pole: ghi chú

## Quy ước

- `q = [x, θ]`, `θ = 0` treo xuống, `θ = π` thẳng đứng. Tâm khối con lắc ở `(x + l sin θ, −l cos θ)`.
- Trục khớp `hinge` trong MuJoCo là `−y` để khớp với quy ước trên.
- Mốc thế năng: `V = 0` ở độ cao khớp, nên `V = −m_p g l cos θ`.

## Bước 2: phương trình chuyển động

```
T = ½(m_c + m_p)ẋ² + m_p l cos θ ẋθ̇ + ½(I + m_p l²)θ̇²
M = [[m_c + m_p,      m_p l cos θ],
     [m_p l cos θ,    I + m_p l² ]]
C = [[0, −m_p l sin θ θ̇],
     [0,  0             ]]
g = [0, m_p g l sin θ]
```
Kết quả: sai số so với `mj_fullM` là 2e-17, với `qfrc_bias` là 2e-15, với `qacc` là 2e-14.

Chú ý API: MuJoCo 3.17 dùng `mj_fullM(model, data, dst)`, không còn `data.qM`.

## Bước 3: năng lượng

10 s từ `θ = π − 0.1`, `dt = 0.002`: RK4 tự viết trôi 4e-10 J, Euler tự viết trôi 0.13 J. Quỹ đạo RK4 tự viết trùng quỹ đạo MuJoCo (RK4) tới 2e-11.

Euler trôi vì nó đi theo độ dốc ở đầu bước; với dao động, mỗi bước đẩy trạng thái ra ngoài một chút, nên năng lượng tăng dần. Sai số tổng của Euler là O(dt), của RK4 là O(dt⁴).

Mốc tham khảo: bộ tích phân của MuJoCo, 60 s từ `up_tilt`: `Euler` (semi-implicit) dao động 3.9e-2 J, `RK4` trôi 3.3e-7 J.

## Bước 4: tuyến tính hóa

Tại đỉnh `qd = 0`, nên số hạng `C q̇` (bậc hai theo `q̇`) biến mất: `A = [[0, I], [−M0⁻¹ ∂g/∂q, 0]]`, `B = [0; M0⁻¹ B_u]`.

Trị riêng: `0, 0, ±5.29`. Một trị riêng dương nên hệ không ổn định, thời gian đổ đặc trưng khoảng `1/5.29 ≈ 0.19 s`. Hạng ma trận điều khiển được là 4.

`mjd_transitionFD` không hỗ trợ RK4, nên `A, B` rời rạc được lấy bằng sai phân quanh `mj_step`. Kết quả khớp `expm(A dt)` tới 3e-10.

## Bước 5: LQR

`Q = diag(10, 50, 1, 1)`, `R = 0.1`, `K ≈ [−10, 63.9, −10.8, 12.2]`. LQR rời rạc cho `K` gần như giống, vì `dt` nhỏ.

| Q_θ | R | thời gian hội tụ | max abs(u) |
|---|---|---|---|
| 10 | 1.0 | 2.97 s | 7.5 N |
| 50 | 0.1 | 2.20 s | 12.8 N |
| 200 | 0.01 | 2.36 s | 20.0 N (bão hòa) |

Q lớn hơn hoặc R nhỏ hơn thì lực lớn hơn. Khi lực chạm giới hạn, tăng thêm không làm nhanh hơn nữa.

Vùng hút (lưới `θ − π ∈ [−1.2, 1.2]`, `θ̇ ∈ [−6, 6]`, xe đứng yên ở `x = 0`): 62% điểm thành công. Vùng có dạng dải chéo: lệch về một phía nhưng đang quay về đỉnh thì vẫn cứu được. `ρ = 29` là mức `xᵀSx` lớn nhất mà bên trong toàn điểm thành công.

## Bước 6: swing-up

Dùng năng lượng của riêng con lắc `E = ½Jθ̇² − m_p g l cos θ` (với `J = I + m_p l²`), không tính xe, vì `Ė = −m_p l cos θ θ̇ ẍ` chỉ phụ thuộc `ẍ`. Chọn `ẍ = k_E Ẽ θ̇ cos θ − k_x x − k_ẋ ẋ`, rồi PFL đổi `ẍ` thành lực `u`.

Hệ số `k_E = 5, k_x = 4, k_ẋ = 3`: `Ẽ` còn 0.23% của `E_đỉnh`, xe chạy tối đa 0.9 m, lực lớn nhất 16.8 N.

## Bước 7: ghép

Chuyển sang LQR khi `xᵀSx < ρ`, quay lại swing-up chỉ khi `xᵀSx > 3ρ` (trễ để chống nhảy chế độ). Từ `θ = 0.05`: chuyển chế độ tại t = 1.02 s, vào vùng `|θ − π| < 0.05` từ t = 3.02 s, giữ 17 s, chỉ chuyển chế độ 1 lần.

Nhận xét: con lắc nhẹ (0.2 kg) và lực 20 N khá mạnh, nên swing-up gần như chỉ cần một lần lắc, và LQR bắt con lắc khi còn cách đỉnh khoảng 1.5 rad (lực bão hòa ngắn lúc bắt). Nếu muốn thấy swing-up nhiều nhịp hơn, giảm `ctrlrange` hoặc `k_E`.

## Câu hỏi cần trả lời

_(phần này để bạn tự viết)_

1. Vì sao cart-pole là hệ thiếu actuator (2 bậc tự do, 1 actuator), và điều này liên quan gì tới 6 bậc tự do floating base của G1?
2. PFL collocated cho phép điều khiển trực tiếp biến nào? Vì sao không làm như vậy được với biến còn lại?
3. `Q, R` ảnh hưởng thế nào tới các cực vòng kín?
4. Vùng hút đo được lớn hay nhỏ hơn bạn đoán, và vì sao?
