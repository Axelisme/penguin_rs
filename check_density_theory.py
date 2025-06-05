import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from util.stable_temp import get_stable_point
from util.stastistic import get_env_temps_at_positions
from util.theory.density import TheoreticalEvolution

load_path = os.path.join("data", "N500_T100s_C(True)", "simulation.npz")
save_path = load_path.replace(".npz", "_check_theory2.mp4")
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
x_range = [10, 30]  # Body temperature range
y_range = [-25, 15]  # Environmental temperature range
NUM_X_BINS = 200  # Number of bins for body temperature
NUM_Y_BINS = 500  # Number of bins for environmental temperature


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

    # 計算穩定點
    x_stable, y_stable = get_stable_point(PREFER_TEMP, HEAT_GEN_COEFF, HEAT_E2P_COEFF)
    print(f"Stable point: ({x_stable:.2f}, {y_stable:.2f})")

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

    # 創建 2x2 子圖佈局
    fig, ((ax_y_density, ax_density), (ax_div_j, ax_x_density)) = plt.subplots(
        2, 2, figsize=(20, 16)
    )

    # 右上：2D密度圖
    density_im = ax_density.imshow(
        theory_evolution.density,
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        aspect="auto",
        origin="lower",
        cmap="hot",
        vmin=0,
    )

    # 添加穩定點到密度圖
    ax_density.plot(
        x_stable,
        y_stable,
        "bo",
        markersize=10,
        label="Stable Point",
        markeredgecolor="white",
        markeredgewidth=2,
    )
    ax_density.axvline(
        PREFER_TEMP,
        color="blue",
        linestyle="--",
        alpha=0.7,
        linewidth=2,
        label="Prefer Temp",
    )
    ax_density.set_xlabel("Body Temperature (x)", fontsize=12)
    ax_density.set_ylabel("Environmental Temperature (y)", fontsize=12)
    ax_density.set_title("2D Density Distribution", fontsize=14)
    ax_density.legend()

    # 密度色標
    cbar_density = plt.colorbar(density_im, ax=ax_density, fraction=0.046, pad=0.04)
    cbar_density.set_label("Density", fontsize=11)

    # 左上：y方向密度曲線
    density_sum_over_x_init = np.sum(theory_evolution.density, axis=1) * (xs[1] - xs[0])
    (y_density_line,) = ax_y_density.plot(
        density_sum_over_x_init, ys, "r-", linewidth=2
    )
    ax_y_density.set_xlabel("Integrated Density over x", fontsize=12)
    ax_y_density.set_ylabel("Environmental Temperature (y)", fontsize=12)
    ax_y_density.set_title("Y-Direction Density Profile", fontsize=14)
    ax_y_density.set_ylim(y_range)
    ax_y_density.grid(True, alpha=0.3)

    # 右下：x方向密度曲線
    density_sum_over_y_init = np.sum(theory_evolution.density, axis=0) * (ys[1] - ys[0])
    (x_density_line,) = ax_x_density.plot(
        xs, density_sum_over_y_init, "g-", linewidth=2
    )
    ax_x_density.axvline(
        PREFER_TEMP,
        color="blue",
        linestyle="--",
        alpha=0.7,
        linewidth=2,
        label="Prefer Temp",
    )
    ax_x_density.axvline(
        x_stable,
        color="red",
        linestyle="--",
        alpha=0.7,
        linewidth=2,
        label="Stable Point X",
    )
    ax_x_density.set_xlabel("Body Temperature (x)", fontsize=12)
    ax_x_density.set_ylabel("Integrated Density over y", fontsize=12)
    ax_x_density.set_title("X-Direction Density Profile", fontsize=14)
    ax_x_density.set_xlim(x_range)
    ax_x_density.grid(True, alpha=0.3)
    ax_x_density.legend()

    # 左下：散度對x軸積分的曲線圖
    # 計算初始散度
    vx = theory_evolution.dxdt(
        theory_evolution.density, theory_evolution.X, theory_evolution.Y
    )
    vy = theory_evolution.dydt(
        theory_evolution.density, theory_evolution.X, theory_evolution.Y
    )
    jx = theory_evolution.density * vx
    jy = theory_evolution.density * vy
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    div_j = np.gradient(jx, dx, axis=1) + np.gradient(jy, dy, axis=0)

    # 計算div_j對x軸的積分
    div_j_integrated_over_x = np.sum(div_j, axis=1) * dx

    (div_j_line,) = ax_div_j.plot(div_j_integrated_over_x, ys, "purple", linewidth=2)
    ax_div_j.set_xlabel("Integrated div_j over x", fontsize=12)
    ax_div_j.set_ylabel("Environmental Temperature (y)", fontsize=12)
    ax_div_j.set_title("Divergence Integrated over X", fontsize=14)
    ax_div_j.set_ylim(y_range)
    ax_div_j.grid(True, alpha=0.3)

    # 統計資訊文字
    stats_text = ax_density.text(
        0.02,
        0.98,
        "",
        transform=ax_density.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    def update(frame):
        """動畫更新函數"""
        # 演化理論系統
        for _ in range(STEPS_PER_FRAME):
            theory_evolution.step(DT)

        current_t = (frame + 1) * STEPS_PER_FRAME * DT
        current_density = theory_evolution.density

        # 更新2D密度圖
        density_im.set_array(current_density)
        density_im.set_clim(
            vmax=np.max(current_density) if np.max(current_density) > 0 else 1e-9
        )

        # 更新y方向密度曲線
        density_sum_over_x = np.sum(current_density, axis=1) * (xs[1] - xs[0])
        y_density_line.set_xdata(density_sum_over_x)
        ax_y_density.set_xlim(
            0,
            np.max(density_sum_over_x) * 1.1
            if np.max(density_sum_over_x) > 0
            else 1e-9,
        )

        # 更新x方向密度曲線
        density_sum_over_y = np.sum(current_density, axis=0) * (ys[1] - ys[0])
        x_density_line.set_ydata(density_sum_over_y)
        ax_x_density.set_ylim(
            0,
            np.max(density_sum_over_y) * 1.1
            if np.max(density_sum_over_y) > 0
            else 1e-9,
        )

        # 更新散度積分曲線
        vx = theory_evolution.dxdt(
            current_density, theory_evolution.X, theory_evolution.Y
        )
        vy = theory_evolution.dydt(
            current_density, theory_evolution.X, theory_evolution.Y
        )
        jx = current_density * vx
        jy = current_density * vy

        correction_term = np.mean(jy, axis=1)
        correction_term -= correction_term[0]

        jy = jy - correction_term[:, None]

        div_j = np.gradient(jx, dx, axis=1) + np.gradient(jy, dy, axis=0)

        # 計算div_j對x軸的積分
        div_j_integrated_over_x = np.sum(div_j, axis=1) * dx
        div_j_line.set_xdata(div_j_integrated_over_x)

        # 動態調整x軸範圍
        if len(div_j_integrated_over_x) > 0:
            x_min, x_max = (
                np.min(div_j_integrated_over_x),
                np.max(div_j_integrated_over_x),
            )
            x_range_div = x_max - x_min
            if x_range_div > 0:
                ax_div_j.set_xlim(x_min - 0.1 * x_range_div, x_max + 0.1 * x_range_div)
            else:
                ax_div_j.set_xlim(-1e-9, 1e-9)

        # 計算統計資訊
        total_mass = np.sum(current_density) * (xs[1] - xs[0]) * (ys[1] - ys[0])
        max_div_j_integrated = np.max(np.abs(div_j_integrated_over_x))
        mean_div_j_integrated = np.mean(np.abs(div_j_integrated_over_x))

        stats_str = (
            f"Time: {current_t:.1f}s\n"
            f"Total Mass: {total_mass:.3f}\n"
            f"Max |∫div_j dx|: {max_div_j_integrated:.2e}\n"
            f"Mean |∫div_j dx|: {mean_div_j_integrated:.2e}"
        )
        stats_text.set_text(stats_str)

        print(
            f"\rFrame {frame + 1}/{TOTAL_FRAMES}, Time: {current_t:.1f}s, "
            f"Mass: {total_mass:.3f}, Max|∫div_j dx|: {max_div_j_integrated:.2e}",
            end="",
        )

        return [
            density_im,
            y_density_line,
            x_density_line,
            div_j_line,
            stats_text,
        ]

    print("Starting density evolution animation...")
    print(f"Parameters: dt={DT}, frames={TOTAL_FRAMES}, total_time={SIM_TIME}s")
    print(f"Grid size: {NUM_X_BINS}x{NUM_Y_BINS}")
    print(f"Stable point: ({x_stable:.2f}, {y_stable:.2f})")

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

    return ani


if __name__ == "__main__":
    # 當直接運行此檔案時，顯示密度演化動畫

    ani = create_density_evolution_animation()

    if save_path is not None:
        ani.save(save_path, writer="ffmpeg", fps=FRAMES_PER_SECOND)
    else:
        plt.show()
