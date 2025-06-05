import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from util.stastistic import get_env_temps_at_positions
from util.theory.density import TheoreticalEvolution

load_path = os.path.join("data", "N500_T100s_C(True)", "simulation.npz")
save_path = load_path.replace(".npz", "_theory2.mp4")
# save_path = None

# Default parameters (will be overridden if loading from file)
NUM_PENGUINS = 500
PENGUIN_MOVE_FACTOR = 0.05
PENGUIN_RADIUS = 0.1
HEAT_GEN_COEFF = 0.15
HEAT_E2P_COEFF = 0.01
PREFER_TEMP = 20.0
BOX_SIZE = 9.0
TEMP_ROOM = -30.0

# 模擬參數
SIM_TIME = 100.0
DT = 1e-4
TOTAL_STEPS = int(SIM_TIME / DT)
FRAMES_PER_SECOND = 20
STEPS_PER_FRAME = max(1, int(1 / (FRAMES_PER_SECOND * DT)))
TOTAL_FRAMES = int(TOTAL_STEPS / STEPS_PER_FRAME)


# 設置繪圖範圍和密度網格參數
x_range = [15, 25]  # Body temperature range
y_range = [-25, 15]  # Environmental temperature range
NUM_X_BINS = 180  # Number of bins for body temperature
NUM_Y_BINS = 180  # Number of bins for environmental temperature


def grad_func(y):
    grad = 25.57 * np.exp(-((y + 3.63) ** 2) / (2 * 9.47**2)) - 5.11
    return np.clip(grad, 1e-6, None)


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
        # 使用預設的隨機初始化
        init_body_temps = np.random.normal(19, 2.0, NUM_PENGUINS)
        init_env_temps = np.random.normal(-5, 5.0, NUM_PENGUINS)
        return init_body_temps, init_env_temps

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


def create_initial_density(
    body_temps, env_temps, xs, ys, penguin_contribution_radius=1.0
):
    """
    創建初始密度分佈
    每隻企鵝貢獻一個半徑為 penguin_contribution_radius 的常態分佈
    """
    density = np.zeros((len(ys), len(xs)))

    # 每隻企鵝的貢獻
    for body_temp, env_temp in zip(body_temps, env_temps):
        # 為每隻企鵝創建一個常態分佈
        x_contrib = np.exp(-0.5 * ((xs - body_temp) / penguin_contribution_radius) ** 2)
        y_contrib = np.exp(-0.5 * ((ys - env_temp) / penguin_contribution_radius) ** 2)

        # 外積得到二維分佈
        penguin_density = np.outer(y_contrib, x_contrib)

        # 歸一化（使每隻企鵝的總貢獻為1）
        penguin_density /= np.sum(penguin_density) * (xs[1] - xs[0]) * (ys[1] - ys[0])

        density += penguin_density

    return density


def create_density_evolution_animation():
    """創建基於密度的理論演化動畫"""

    # 載入模擬資料
    init_body_temps, init_env_temps = load_simulation_data(load_path)

    # 創建座標網格
    xs = np.linspace(x_range[0], x_range[1], NUM_X_BINS)
    ys = np.linspace(y_range[0], y_range[1], NUM_Y_BINS)

    # 創建初始密度分佈
    print("Creating initial density distribution...")
    init_density = create_initial_density(init_body_temps, init_env_temps, xs, ys)

    # 創建理論演化系統
    theory_evolution = TheoreticalEvolution(
        init_density,
        xs,
        ys,
        grad_func,
        HEAT_GEN_COEFF,
        HEAT_E2P_COEFF,
        PENGUIN_MOVE_FACTOR,
        PREFER_TEMP,
    )

    # 創建圖形
    fig, (ax2, ax1) = plt.subplots(
        1, 2, figsize=(20, 8), gridspec_kw={"width_ratios": [1, 3]}
    )  # ax2 for y-density, ax1 for 2D density

    # 初始密度圖 (ax1 - 右側)
    density_im = ax1.imshow(
        theory_evolution.density,
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        aspect="auto",
        origin="lower",
        cmap="hot",
        vmin=0,
    )

    # 添加穩定點到 ax1
    ax1.axvline(
        PREFER_TEMP,
        color="blue",
        linestyle="--",
        alpha=0.7,
        linewidth=2,
        label="Prefer Temp",
    )

    ax1.set_xlabel("Body Temperature (x)", fontsize=12)
    ax1.set_ylabel("Environmental Temperature (y)", fontsize=12)
    ax1.set_title("Density Evolution (X-Y Plane)", fontsize=14)
    ax1.legend()

    # 密度色標 for ax1
    cbar1 = plt.colorbar(density_im, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Density", fontsize=11)

    # 初始 y-密度圖 (ax2 - 左側)
    density_sum_over_x_init = np.sum(theory_evolution.density, axis=1) * (
        xs[1] - xs[0]
    )  # Integrate over x
    (y_density_line,) = ax2.plot(density_sum_over_x_init, ys)
    ax2.set_xlabel("Integrated Density over x", fontsize=12)
    ax2.set_ylabel("Environmental Temperature (y)", fontsize=12)
    ax2.set_title("Density vs. Y", fontsize=14)
    ax2.set_ylim(y_range)
    ax2.grid(True)

    # 統計資訊文字 (放在 ax1 上方)
    stats_text = ax1.text(  # Use ax1.text instead of fig.text for blit compatibility
        0.5,  # Centered horizontally in ax1
        1.05,  # Above ax1
        "",
        transform=ax1.transAxes,  # Relative to ax1
        verticalalignment="bottom",
        horizontalalignment="center",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    def update(frame):
        """動畫更新函數"""
        # 演化理論系統
        for _ in range(STEPS_PER_FRAME):
            theory_evolution.step(DT)

        current_t = (frame + 1) * STEPS_PER_FRAME * DT
        current_density = theory_evolution.density

        # 更新密度圖 (ax1)
        density_im.set_array(current_density)
        density_im.set_clim(
            vmax=np.max(current_density) if np.max(current_density) > 0 else 1e-9
        )

        # 更新 y-密度圖 (ax2)
        # Integrate over x
        density_sum_over_x = np.sum(current_density, axis=1) * (xs[1] - xs[0])
        y_density_line.set_xdata(density_sum_over_x)
        ax2.set_xlim(
            0,
            np.max(density_sum_over_x) * 1.1
            if np.max(density_sum_over_x) > 0
            else 1e-9,
        )

        # 計算統計資訊
        total_mass = (
            np.sum(current_density) * (xs[1] - xs[0]) * (ys[1] - ys[0])
        )  # Approximate integral
        stats_str = f"Time: {current_t:.1f}s\nTotal Mass: {total_mass:.3f}"
        stats_text.set_text(stats_str)

        print(
            f"\rFrame {frame + 1}/{TOTAL_FRAMES}, Time: {current_t:.1f}s, Total Mass: {total_mass:.3f}",
            end="",
        )

        return [
            density_im,
            y_density_line,
            stats_text,
        ]  # Only return artist objects, not axes

    print("Starting density evolution animation...")
    print(f"Parameters: dt={DT}, frames={TOTAL_FRAMES}, total_time={SIM_TIME}s")
    print(f"Grid size: {NUM_X_BINS}x{NUM_Y_BINS}")

    # 創建動畫
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=TOTAL_FRAMES,
        interval=1000 / FRAMES_PER_SECOND,
        blit=True,  # Changed back to True with proper artist handling
        repeat=True,
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.90]
    )  # Adjusted rect to accommodate ax1-relative text

    return ani


if __name__ == "__main__":
    # 當直接運行此檔案時，顯示密度演化動畫

    ani = create_density_evolution_animation()

    if save_path is not None:
        ani.save(save_path, writer="ffmpeg", fps=FRAMES_PER_SECOND)
    else:
        plt.show()
