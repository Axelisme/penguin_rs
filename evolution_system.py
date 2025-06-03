import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

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


def grad_func(y):
    grad = 25.57 * np.exp(-((y + 3.63) ** 2) / (2 * 9.47**2)) - 5.11
    return np.clip(grad, 0.0, None)


def load_simulation_data(filename="penguin_simulation_data.npz"):
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


def get_env_temps_at_positions(positions, air_temp_grid, box_size):
    """Get environmental temperature at each penguin position using bilinear interpolation"""
    num_grid = air_temp_grid.shape[0]
    env_temps = []

    for pos in positions:
        # Convert position to continuous grid coordinates
        x_grid = (pos[0] / box_size) * num_grid
        y_grid = (pos[1] / box_size) * num_grid

        # Get integer parts and fractional parts for bilinear interpolation
        x0 = int(np.floor(x_grid)) % num_grid
        y0 = int(np.floor(y_grid)) % num_grid
        x1 = (x0 + 1) % num_grid
        y1 = (y0 + 1) % num_grid

        # Fractional parts
        fx = x_grid - np.floor(x_grid)
        fy = y_grid - np.floor(y_grid)

        # Bilinear interpolation
        temp = (
            air_temp_grid[x0, y0] * (1 - fx) * (1 - fy)
            + air_temp_grid[x1, y0] * fx * (1 - fy)
            + air_temp_grid[x0, y1] * (1 - fx) * fy
            + air_temp_grid[x1, y1] * fx * fy
        )
        env_temps.append(temp)
    return np.array(env_temps)


class TheoreticalEvolution:
    """理論演化系統模擬器"""

    def __init__(self, init_x, init_y):
        # 初始化企鵝的理論狀態 (x=body_temp, y=env_temp)
        self.x = init_x
        self.y = init_y

    def dxdt(self, x, y):
        """體溫演化方程: dx/dt = HEAT_GEN_COEFF - HEAT_E2P_COEFF*(x - y)"""
        return HEAT_GEN_COEFF - HEAT_E2P_COEFF * (x - y)

    def dydt(self, x, y):
        """環境溫度演化方程: dy/dt = -PENGUIN_MOVE_FACTOR * density(y) * (x - PREFER_TEMP) * grad_func(y)"""
        return -PENGUIN_MOVE_FACTOR * (x - PREFER_TEMP) * grad_func(y) ** 2

    def step(self, dt):
        """使用歐拉方法演化一步"""
        dx = self.dxdt(self.x, self.y)
        dy = self.dydt(self.x, self.y)

        # add noise
        dx += np.random.normal(0, 0.1, self.x.shape)
        dy += np.random.normal(0, 0.1, self.y.shape)

        self.x += dx * dt
        self.y += dy * dt

    def get_vector_field(self, x_grid, y_grid):
        """計算向量場用於繪製"""
        X, Y = np.meshgrid(x_grid, y_grid)
        DX = self.dxdt(X, Y)
        DY = self.dydt(X, Y)
        return X, Y, DX, DY

    def get_state(self):
        """獲取當前狀態"""
        return self.x.copy(), self.y.copy()


def get_stable_point():
    """計算理論穩定點"""
    # dx/dt = 0: HEAT_GEN_COEFF - HEAT_E2P_COEFF*(x - y) = 0
    # 當 x = PREFER_TEMP 時，dy/dt = 0
    x_stable = PREFER_TEMP
    y_stable = PREFER_TEMP - HEAT_GEN_COEFF / HEAT_E2P_COEFF
    return x_stable, y_stable


def create_theory_animation(load_file=None):
    """創建理論演化動畫

    Parameters:
    -----------
    load_file : str, optional
        要載入的模擬資料檔案名稱。如果為 None，使用預設檔名 "penguin_simulation_data.npz"
    """

    # 嘗試載入模擬資料
    if load_file is None:
        init_body_temps, init_env_temps = load_simulation_data()
    else:
        init_body_temps, init_env_temps = load_simulation_data(load_file)

    # 模擬參數
    SIM_TIME = 100.0
    DT = 0.01
    TOTAL_STEPS = int(SIM_TIME / DT)
    FRAMES_PER_SECOND = 20
    STEPS_PER_FRAME = max(1, int(1 / (FRAMES_PER_SECOND * DT)))
    TOTAL_FRAMES = int(TOTAL_STEPS / STEPS_PER_FRAME)

    # 計算穩定點
    x_stable, y_stable = get_stable_point()
    print(f"Stable point: ({x_stable:.2f}, {y_stable:.2f})")

    # 設定初始值
    if init_body_temps is not None and init_env_temps is not None:
        # 使用載入的最後一幀數據作為初始值
        init_x = init_body_temps.copy()
        init_y = init_env_temps.copy()
        print("Using loaded simulation data as initial conditions")
    else:
        # 使用隨機初始值
        init_x = np.random.normal(19, 2.0, NUM_PENGUINS)
        init_y = np.random.normal(-5, 5.0, NUM_PENGUINS)
        print("Using random initial conditions")

    # 創建理論演化系統
    theory_evolution = TheoreticalEvolution(init_x, init_y)

    # 設置繪圖範圍

    x_range = [17.5, 22.5]
    y_range = [-20, 15]

    env_temps = np.linspace(*y_range, 100)
    grad_range = grad_func(env_temps)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(env_temps, grad_range**2)
    plt.show()
    # exit()

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
    x_penguins, y_penguins = theory_evolution.get_state()
    scatter = ax.scatter(
        x_penguins,
        y_penguins,
        c="red",
        s=12,
        alpha=0.8,
        animated=True,
        edgecolors="darkred",
        linewidth=0.5,
    )

    # 添加等勢線和穩定點
    ax.plot(
        x_stable,
        y_stable,
        "ro",
        markersize=12,
        label="Stable Point",
        markeredgecolor="darkred",
        markeredgewidth=2,
    )
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
        x_penguins, y_penguins = theory_evolution.get_state()

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
            f"Distance to Stable: {np.sqrt((x_mean - x_stable) ** 2 + (y_mean - y_stable) ** 2):.2f}"
        )

        stats_text.set_text(stats_str)

        print(f"\rFrame {frame + 1}/{TOTAL_FRAMES}, Time: {current_t:.1f}s", end="")

        return [scatter, stats_text]

    print("Starting theoretical evolution animation...")
    print(f"Parameters: dt={DT}, frames={TOTAL_FRAMES}, total_time={SIM_TIME}s")
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
    plt.show()

    print("\nAnimation complete!")

    return ani


if __name__ == "__main__":
    # 當直接運行此檔案時，顯示理論演化動畫
    create_theory_animation()
