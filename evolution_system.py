import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from util.stastistic import get_env_temps_at_positions
from util.theory.particle import TheoreticalEvolution

load_path = os.path.join("data", "N500_T100s_C(False)", "simulation.npz")
save_path = load_path.replace(".npz", "_theory1.mp4")
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


# def grad_func(y):
#     grad = 25.57 * np.exp(-((y + 3.63) ** 2) / (2 * 9.47**2)) - 5.11
#     return np.clip(grad, 0.0, None)


def grad_func(y):
    # -5.02714357e-04 -1.76880183e-02 -2.31417227e-01 -1.83132979e-01, 2.77934649e+01]
    a = -5.02714357e-04
    b = -1.76880183e-02
    c = -2.31417227e-01
    d = -1.83132979e-01
    e = 2.77934649e01
    grad = a * y**4 + b * y**3 + c * y**2 + d * y + e
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


def create_theory_animation():
    """創建理論演化動畫

    Parameters:
    -----------
    load_file : str, optional
        要載入的模擬資料檔案名稱。如果為 None, 使用預設檔名 "penguin_simulation_data.npz"
    """

    init_x, init_y = load_simulation_data(load_path)

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

    # 創建圖形
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # === 繪製向量場 ===
    # 創建向量場網格
    x_vec = np.linspace(x_range[0], x_range[1], 25)
    y_vec = np.linspace(y_range[0], y_range[1], 25)
    X, Y, DX, DY = theory_evolution.get_vector_field(x_vec, y_vec)

    # 先歸一化方向，然後重新縮放長度
    DX_norm = DX
    DY_norm = DY

    # 繪製向量場
    quiver = ax.quiver(
        X,
        Y,
        DX_norm,
        DY_norm,
        cmap="viridis",
        alpha=0.6,
        scale_units="xy",  # 使用資料座標系
        width=0.003,
    )

    # === 繪製企鵝演化 ===
    scatter = ax.scatter(
        theory_evolution.x,
        theory_evolution.y,
        c="red",
        s=12,
        alpha=0.8,
        animated=True,
        edgecolors="darkred",
        linewidth=0.5,
    )

    # 添加等勢線和穩定點
    ax.axvline(
        PREFER_TEMP,
        color="blue",
        linestyle="--",
        alpha=0.7,
        linewidth=2,
        label="Prefer Temp",
    )
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)

    # 添加零增長線
    x_line = np.linspace(x_range[0], x_range[1], 100)
    y_zero_dx = x_line - HEAT_GEN_COEFF / HEAT_E2P_COEFF  # dx/dt = 0 的線
    ax.plot(
        x_line,
        y_zero_dx,
        "orange",
        linestyle="--",
        alpha=0.8,
        linewidth=2,
        label="dx/dt = 0",
    )

    ax.set_xlabel("Body Temperature (x)", fontsize=12)
    ax.set_ylabel("Environmental Temperature (y)", fontsize=12)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f"Theoretical Vector Field with Penguin Evolution (N={NUM_PENGUINS})",
        fontsize=14,
    )
    ax.legend(loc="upper right")

    # 添加色標
    cbar = plt.colorbar(quiver, ax=ax)
    cbar.set_label("Vector Magnitude", fontsize=11)

    # 統計資訊文字
    stats_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
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
        x_penguins = theory_evolution.x
        y_penguins = theory_evolution.y

        # 更新散點圖
        scatter.set_offsets(np.column_stack([x_penguins, y_penguins]))

        # 計算統計資訊
        x_mean, x_std = np.mean(x_penguins), np.std(x_penguins)
        y_mean, y_std = np.mean(y_penguins), np.std(y_penguins)
        # ax.set_xlim(x_mean - 10 * x_std, x_mean + 10 * x_std)
        # ax.set_ylim(y_mean - 3 * y_std, y_mean + 10 * y_std)

        stats_str = (
            f"Time: {current_t:.1f}s\n"
            f"Body Temp: {x_mean:.2f}±{x_std:.2f}\n"
            f"Env Temp: {y_mean:.2f}±{y_std:.2f}\n"
        )

        stats_text.set_text(stats_str)

        print(f"\rFrame {frame + 1}/{TOTAL_FRAMES}, Time: {current_t:.1f}s", end="")

        return [scatter, stats_text]

    print("Starting theoretical evolution animation...")
    print(f"Parameters: dt={DT}, frames={TOTAL_FRAMES}, total_time={SIM_TIME}s")

    # 創建動畫
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=TOTAL_FRAMES,
        interval=1000 / FRAMES_PER_SECOND,
        blit=True,
        repeat=True,
    )

    plt.tight_layout()

    print("\nAnimation complete!")

    return ani


if __name__ == "__main__":
    # 當直接運行此檔案時，顯示理論演化動畫

    ani = create_theory_animation()

    if save_path is not None:
        ani.save(save_path, writer="ffmpeg", fps=FRAMES_PER_SECOND)
    else:
        plt.show()
