from math import pi, sqrt

import numpy as np


def sovle_temp_eqution(C, h0, r, cutoff_G=10) -> float:
    """
    計算二維最密堆積晶格下微分方程 T - C∇²T = h(r)
    在原點處的解，假設 h(r) = h0 * Σ δ(r - R)

    參數：
        C: 擴散係數
        h0: 源項強度 (物理意義上的單個點源強度)
        r: 晶格常數（距離）
        cutoff_G: 最多考慮 reciprocal vector G 的模長限制（實際是 index 範圍）

    回傳：
        T(0): 在原點的穩定解值 (考慮了正確的歸一化因子)
    """
    # Reciprocal lattice vectors (analytically derived)
    b1 = (2 * pi / r) * np.array([1, -1 / sqrt(3)])
    b2 = (2 * pi / r) * np.array([0, 2 / sqrt(3)])

    # k-space summation part
    sum_G_term = 0.0
    for m in range(-cutoff_G, cutoff_G + 1):
        for n in range(-cutoff_G, cutoff_G + 1):
            G = m * b1 + n * b2
            G2 = np.dot(G, G)
            # The term for G=0 is 1/(1+0) = 1, which is correctly handled.
            sum_G_term += 1 / (1 + C * G2)

    A_cell = (sqrt(3) / 2) * r**2

    # Physical solution T(0) = (h0 / A_cell) * sum_G_term
    return (h0 / A_cell) * sum_G_term


def calculate_stable_temp(A, B, C, D, r, epsilon, t_room) -> float:
    """
    計算以下方程的解：
    $$
    0 = C * ∇²T + A*D/B * \sum_{R} N(R; 0, r) - \epsilon * (T - T_room)
    $$
    其中 N(R; 0, r) 是高度為1的 Gaussian 分布。
    """

    # 計算高斯分佈的面積作為源項強度
    h0 = A * D / B * 2 * pi * r**2

    return t_room + sovle_temp_eqution(C / epsilon, h0 / epsilon, 2 * r, cutoff_G=50)


if __name__ == "__main__":
    pass
