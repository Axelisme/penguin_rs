import math

import scipy.special as sp


def calculate_T0(epsilon, C, r):
    """
    根據給定參數計算 T 在原點 (r=0) 的值。

    公式: T(0) = (r^2 / (2C)) * exp(epsilon*r^2 / (2C)) * E_1(epsilon*r^2 / (2C))

    參數：
    epsilon (float): 微分方程中的常數 epsilon, 必須是正數。
    C (float): 微分方程中的常數 C, 必須是正數。
    r (float): D(r) 中高斯分佈的標準差參數 r, 必須是正數。

    返回：
    float: T(0) 的數值解。
    """

    arg_E1 = (epsilon * r**2) / (2 * C)

    # 檢查 E1 函數的參數是否極小，防止潛在的數值問題，儘管理論上它不會是0或負數
    if arg_E1 < 1e-10:  # 如果接近0，E1會非常大
        # 這裡可以根據需要添加警告或更精確的處理，但對於正常物理情況應為正值
        print(
            f"Warning: Argument for E_1 is very small ({arg_E1}), result might be very large."
        )

    e1_value = sp.exp1(arg_E1)

    return (r**2 / (2 * C)) * math.exp(arg_E1) * e1_value


def calculate_stable_temp(A, B, C, D, r, epsilon, t_room) -> float:
    r"""
    計算以下方程的解：
    $$
    0 = C * ∇²T + A*D/B * \sum_{R} N(R; 0, r) - \epsilon * (T - T_room)
    $$
    其中 N(R; 0, r) 是高度為1的 Gaussian 分布。
    """

    return A * D / B * calculate_T0(epsilon, C, r) + t_room
