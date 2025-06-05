import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from util.stastistic import get_env_temps_at_positions
from util.theory.particle import TheoreticalEvolution

load_path = os.path.join("data", "N500_T100s_C(True)", "simulation.npz")
save_path = load_path.replace(".npz", "_check_theory1.mp4")
save_path = None

# Default parameters (will be overridden if loading from file)
NUM_PENGUINS = 500
PENGUIN_MOVE_FACTOR = 0.05
PENGUIN_RADIUS = 0.1
HEAT_GEN_COEFF = 0.15
HEAT_P2E_COEFF = 1.0
HEAT_E2P_COEFF = 0.01
PREFER_TEMP = 20.0
BOX_SIZE = 9.0
DIFFUSION_COEFF = 0.4
DECAY_COEFF = 0.4
TEMP_ROOM = -30.0
COLLISION_STRENGTH = 10.0  # 碰撞排斥力强度

# 模擬參數
SIM_TIME = 100.0
DT = 0.01
TOTAL_STEPS = int(SIM_TIME / DT)
FRAMES_PER_SECOND = 20
STEPS_PER_FRAME = max(1, int(1 / (FRAMES_PER_SECOND * DT)))
TOTAL_FRAMES = int(TOTAL_STEPS / STEPS_PER_FRAME)


# 設置繪圖範圍
x_range = [17.5, 22.5]
y_range = [-20, 15]


def grad_func(y):
    grad = 25.57 * np.exp(-((y + 3.63) ** 2) / (2 * 9.47**2)) - 5.11
    return np.clip(grad, 0.0, None)


def load_simulation_data(filename):
    """載入模擬資料並返回最後一幀的數據作為初始值"""
    global \
        NUM_PENGUINS, \
        PENGUIN_MOVE_FACTOR, \
        HEAT_GEN_COEFF, \
        HEAT_E2P_COEFF, \
        PREFER_TEMP, \
        TEMP_ROOM, \
        BOX_SIZE

    if not os.path.exists(filename):
        print(f"Warning: {filename} not found. Using default random initialization.")
        return None, None

    print(f"Loading simulation data from {filename}...")
    npz = np.load(filename, allow_pickle=True)

    positions = npz["positions"]  # shape: (frames, N, 2)
    body_temps = npz["body_temps"]  # shape: (frames, N)
    air_temps = npz["air_temps"]  # shape: (frames, num_grid, num_grid)
    times = npz["times"]  # shape: (frames,)
    params = npz["params"].item()  # dict

    # Update global parameters from loaded data
    NUM_PENGUINS = positions.shape[1]
    PENGUIN_MOVE_FACTOR = params["PENGUIN_MOVE_FACTOR"]
    HEAT_GEN_COEFF = params["HEAT_GEN_COEFF"]
    HEAT_E2P_COEFF = params["HEAT_E2P_COEFF"]
    PREFER_TEMP = params["PREFER_TEMP"]
    TEMP_ROOM = params["TEMP_ROOM"]
    BOX_SIZE = params["BOX_SIZE"]

    # Get last frame data as initial values
    last_frame_idx = -1
    init_positions = positions[last_frame_idx]  # shape: (N, 2)
    init_body_temps = body_temps[last_frame_idx]  # shape: (N,)

    # For environment temperature, we need to sample from the air temperature grid
    # at penguin positions using interpolation
    init_env_temps = get_env_temps_at_positions(
        init_positions, air_temps[last_frame_idx], BOX_SIZE
    )

    print(
        f"Loaded data: {NUM_PENGUINS} penguins, final time: {times[last_frame_idx]:.2f}s"
    )
    print(
        f"Initial body temp range: [{np.min(init_body_temps):.2f}, {np.max(init_body_temps):.2f}]"
    )
    print(
        f"Initial env temp range: [{np.min(init_env_temps):.2f}, {np.max(init_env_temps):.2f}]"
    )

    return init_body_temps, init_env_temps


def calculate_pi_distribution(y_values, sigma=1.0):
    """計算pi分佈"""
    yi = y_values[:, None]
    yj = y_values[None, :]

    # pi = \sum_j N(y_i, y_j, sigma)
    pi = np.sum(np.exp(-0.5 * ((yi - yj) / sigma) ** 2), axis=1)

    return pi


def create_particle_theory_animation():
    """創建粒子理論驗證動畫，顯示即時粒子分佈和pi分佈"""

    init_x, init_y = load_simulation_data(load_path)

    if init_x is None or init_y is None:
        # 使用預設初始化
        init_x = np.random.normal(20.0, 1.0, NUM_PENGUINS)
        init_y = np.random.normal(-5.0, 5.0, NUM_PENGUINS)

    # 創建理論演化系統
    theory_evolution = TheoreticalEvolution(
        init_x,
        init_y,
        grad_func,
        HEAT_GEN_COEFF,
        HEAT_E2P_COEFF,
        PENGUIN_MOVE_FACTOR,
        PREFER_TEMP,
    )

    # 創建子圖
    fig = plt.figure(figsize=(16, 10))

    # 主要散點圖 (x-y空間中的粒子分佈)
    ax1 = plt.subplot(2, 3, (1, 4))
    scatter = ax1.scatter(
        theory_evolution.x,
        theory_evolution.y,
        c="red",
        s=12,
        alpha=0.8,
        edgecolors="darkred",
        linewidth=0.5,
    )

    ax1.set_xlabel("Body Temperature (x)", fontsize=12)
    ax1.set_ylabel("Environmental Temperature (y)", fontsize=12)
    ax1.set_xlim(x_range)
    ax1.set_ylim(y_range)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Particle Distribution in (x, y) Space", fontsize=14)

    # 添加偏好溫度線
    ax1.axvline(
        PREFER_TEMP,
        color="blue",
        linestyle="--",
        alpha=0.7,
        linewidth=2,
        label="Prefer Temp",
    )
    ax1.legend()

    # x方向直方圖
    ax2 = plt.subplot(2, 3, 2)
    x_hist_bins = np.linspace(x_range[0], x_range[1], 30)
    x_hist_n, x_hist_bins_edges, x_hist_patches = ax2.hist(
        theory_evolution.x, bins=x_hist_bins, alpha=0.7, color="blue", edgecolor="black"
    )
    ax2.set_xlabel("Body Temperature (x)", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("X Distribution", fontsize=12)
    ax2.grid(True, alpha=0.3)

    # y方向直方圖
    ax3 = plt.subplot(2, 3, 3)
    y_hist_bins = np.linspace(y_range[0], y_range[1], 30)
    y_hist_n, y_hist_bins_edges, y_hist_patches = ax3.hist(
        theory_evolution.y,
        bins=y_hist_bins,
        alpha=0.7,
        color="green",
        edgecolor="black",
    )
    ax3.set_xlabel("Environmental Temperature (y)", fontsize=12)
    ax3.set_ylabel("Count", fontsize=12)
    ax3.set_title("Y Distribution", fontsize=12)
    ax3.grid(True, alpha=0.3)

    # pi分佈圖
    ax4 = plt.subplot(2, 3, 5)
    pi_values = calculate_pi_distribution(theory_evolution.y)
    pi_scatter = ax4.scatter(
        theory_evolution.y,
        pi_values,
        c="orange",
        s=12,
        alpha=0.8,
        edgecolors="darkorange",
        linewidth=0.5,
    )
    ax4.set_xlabel("Environmental Temperature (y)", fontsize=12)
    ax4.set_ylabel("Pi Value", fontsize=12)
    ax4.set_title("Pi vs Y Distribution", fontsize=12)
    ax4.grid(True, alpha=0.3)

    # 統計資訊
    ax5 = plt.subplot(2, 3, 6)
    ax5.axis("off")
    stats_text = ax5.text(
        0.1,
        0.9,
        "",
        transform=ax5.transAxes,
        verticalalignment="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    def update(frame):
        """動畫更新函數"""
        # 演化理論系統
        for _ in range(STEPS_PER_FRAME):
            theory_evolution.step(DT)

        current_t = (frame + 1) * STEPS_PER_FRAME * DT

        # 獲取當前狀態
        x_penguins = theory_evolution.x.copy()
        y_penguins = theory_evolution.y.copy()

        # 計算pi分佈
        pi_values = calculate_pi_distribution(y_penguins)

        # 更新主散點圖
        offsets = np.column_stack([x_penguins, y_penguins])
        scatter.set_offsets(offsets)

        # 確保散點圖在正確的範圍內
        if len(x_penguins) > 0:
            ax1.set_ylim(
                min(y_range[0], np.min(y_penguins) - 1),
                max(y_range[1], np.max(y_penguins) + 1),
            )

        # 更新x方向直方圖
        ax2.clear()
        ax2.hist(
            x_penguins, bins=x_hist_bins, alpha=0.7, color="blue", edgecolor="black"
        )
        ax2.set_xlabel("Body Temperature (x)", fontsize=12)
        ax2.set_ylabel("Count", fontsize=12)
        ax2.set_title("X Distribution", fontsize=12)
        ax2.grid(True, alpha=0.3)

        # 更新y方向直方圖
        ax3.clear()
        ax3.hist(
            y_penguins, bins=y_hist_bins, alpha=0.7, color="green", edgecolor="black"
        )
        ax3.set_xlabel("Environmental Temperature (y)", fontsize=12)
        ax3.set_ylabel("Count", fontsize=12)
        ax3.set_title("Y Distribution", fontsize=12)
        ax3.grid(True, alpha=0.3)

        # 更新pi分佈圖
        pi_offsets = np.column_stack([y_penguins, pi_values])
        pi_scatter.set_offsets(pi_offsets)

        # 動態調整軸範圍
        if len(y_penguins) > 0 and len(pi_values) > 0:
            ax4.set_ylim(
                np.min(pi_values) - 0.1 * np.ptp(pi_values),
                np.max(pi_values) + 0.1 * np.ptp(pi_values),
            )

        # 計算統計資訊
        x_mean, x_std = np.mean(x_penguins), np.std(x_penguins)
        y_mean, y_std = np.mean(y_penguins), np.std(y_penguins)
        pi_mean, pi_std = np.mean(pi_values), np.std(pi_values)

        stats_str = (
            f"Time: {current_t:.1f}s\n"
            f"Particles: {len(x_penguins)}\n\n"
            f"Body Temp (x):\n"
            f"  Mean: {x_mean:.2f}\n"
            f"  Std: {x_std:.2f}\n"
            f"  Range: [{np.min(x_penguins):.2f}, {np.max(x_penguins):.2f}]\n\n"
            f"Env Temp (y):\n"
            f"  Mean: {y_mean:.2f}\n"
            f"  Std: {y_std:.2f}\n"
            f"  Range: [{np.min(y_penguins):.2f}, {np.max(y_penguins):.2f}]\n\n"
            f"Pi Distribution:\n"
            f"  Mean: {pi_mean:.2f}\n"
            f"  Std: {pi_std:.2f}\n"
            f"  Range: [{np.min(pi_values):.2f}, {np.max(pi_values):.2f}]"
        )

        stats_text.set_text(stats_str)

        print(f"\rFrame {frame + 1}/{TOTAL_FRAMES}, Time: {current_t:.1f}s", end="")

        return [scatter, pi_scatter, stats_text]

    print("Starting particle theory verification animation...")
    print(f"Parameters: dt={DT}, frames={TOTAL_FRAMES}, total_time={SIM_TIME}s")

    # 創建動畫
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=TOTAL_FRAMES,
        interval=1000 / FRAMES_PER_SECOND,
        blit=False,  # 設為False以便更新多個子圖
        repeat=True,
    )

    plt.tight_layout()

    print("\nAnimation complete!")

    return ani


if __name__ == "__main__":
    # 當直接運行此檔案時，顯示粒子理論驗證動畫

    ani = create_particle_theory_animation()

    if save_path is not None:
        ani.save(save_path, writer="ffmpeg", fps=FRAMES_PER_SECOND)
    else:
        plt.show()
