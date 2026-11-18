tôi đang muốn xây dựng mô hình và điều khiển robot G1, hiện tại hãy làm như sau:
model g1
sử dụng mujoco để mô phỏng và sử dụng pinocchio để giúp tính toán, có thể kèm các thư viện khác nếu cần

robot sẽ được phát triển theo từng chặng

các id của bộ phận nên được đặt theo tên, không để kiểu vd: subtree[1]

### Chặng 1: robot đứng được, chịu được dao động nhỏ

Ràng buộc gốc                 Hàng của A (q̈ | τ | f)   lo              hi
M q̈ + h = Sᵀτ + Jcᵀf          [ M  | -Sᵀ | -Jcᵀ ]       -h              -h
Chân đứng yên                  [ Jf |  0  |  0   ]       -J̇q̇-Kf·Jf·q̇     (như lo)
|τ| ≤ τ_max                    [ 0  |  I  |  0   ]       -τ_max          τ_max
±f_x - μ f_z ≤ 0 (và f_y)      [ 0  |  0  | ±1,-μ ]      -∞              0
f_z ≥ 0                        [ 0  |  0  |  1   ]       0               600
Giới hạn khớp                  [ I  |  0  |  0   ]       q̈_lo            q̈_hi