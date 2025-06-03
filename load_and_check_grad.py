import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from util.stastistic import get_env_temps_at_positions, get_grad_at_positions

load_path = "penguin_simulation_data_no_colli.npz"
save_path = load_path.replace(".npz", "_gradient_analysis.png")


def gaussian_func(x, A, mu, sigma, offset):
    """高斯函數: y = A * exp(-((x - mu)^2) / (2 * sigma^2)) + offset"""
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + offset


def main():
    # 讀取 npz 檔案
    npz = np.load(load_path, allow_pickle=True)

    positions = npz["positions"]  # shape: (frames, N, 2)
    body_temps = npz["body_temps"]  # shape: (frames, N)
    air_temps = npz["air_temps"]  # shape: (frames, num_grid, num_grid)
    times = npz["times"]  # shape: (frames,)
    params = npz["params"].item()  # dict

    print(
        f"Loaded simulation data with {positions.shape[0]} frames, {positions.shape[1]} penguins"
    )
    print(f"Time range: {times[0]:.2f} - {times[-1]:.2f} seconds")

    # 取得參數
    BOX_SIZE = params["BOX_SIZE"]
    NUM_FRAMES, NUM_PENGUINS = body_temps.shape

    # 找到對應於50秒的幀索引
    time_threshold = 50.0
    start_frame_idx = np.searchsorted(times, time_threshold)
    print(f"Filtering out first 50 seconds of data (frames 0-{start_frame_idx - 1})")
    print(f"Using frames {start_frame_idx}-{NUM_FRAMES - 1} for analysis")

    # 收集所有資料點
    penguin_positions_list = []
    penguin_env_temps_list = []
    penguin_body_temps_list = []
    penguin_grad_magnitudes_list = []
    penguin_grad_x_list = []
    penguin_grad_y_list = []

    print("Calculating gradients at penguin positions...")

    # 遍歷每個時間幀 (從50秒後開始)
    for frame_idx in range(start_frame_idx, NUM_FRAMES):
        curr_positions = positions[frame_idx]
        curr_body_temps = body_temps[frame_idx]
        curr_air_temps = air_temps[frame_idx]

        # 獲取每隻企鵝位置的環境溫度
        env_temps_at_positions = get_env_temps_at_positions(
            curr_positions, curr_air_temps, BOX_SIZE
        )

        # 獲取每隻企鵝位置的溫度梯度
        gradients_at_positions = get_grad_at_positions(
            curr_positions, curr_air_temps, BOX_SIZE
        )

        # 計算梯度大小
        grad_magnitudes = np.sqrt(
            gradients_at_positions[:, 0] ** 2 + gradients_at_positions[:, 1] ** 2
        )

        # 收集資料
        for penguin_idx in range(NUM_PENGUINS):
            if not (
                np.isnan(grad_magnitudes[penguin_idx])
                or np.isinf(grad_magnitudes[penguin_idx])
            ):
                penguin_positions_list.append(curr_positions[penguin_idx])
                penguin_env_temps_list.append(env_temps_at_positions[penguin_idx])
                penguin_body_temps_list.append(curr_body_temps[penguin_idx])
                penguin_grad_magnitudes_list.append(grad_magnitudes[penguin_idx])
                penguin_grad_x_list.append(gradients_at_positions[penguin_idx, 0])
                penguin_grad_y_list.append(gradients_at_positions[penguin_idx, 1])

    # 轉換為numpy arrays
    penguin_env_temps = np.array(penguin_env_temps_list)
    penguin_body_temps = np.array(penguin_body_temps_list)
    penguin_grad_magnitudes = np.array(penguin_grad_magnitudes_list)

    print(f"Total data points collected: {len(penguin_env_temps)}")

    # 降採樣資料以提高繪圖效率
    downsample_factor = max(1, len(penguin_env_temps) // 8000)  # 目標約8000個點
    downsample_indices = np.arange(0, len(penguin_env_temps), downsample_factor)

    env_temps_down = penguin_env_temps[downsample_indices]
    body_temps_down = penguin_body_temps[downsample_indices]
    grad_mag_down = penguin_grad_magnitudes[downsample_indices]

    print(f"Downsampled to {len(env_temps_down)} points (factor: {downsample_factor})")

    # 計算溫度區間的平均梯度 (用於擬合) - 只使用-15度以上的資料
    temp_threshold = -15.0
    valid_temp_mask = penguin_env_temps >= temp_threshold
    valid_env_temps = penguin_env_temps[valid_temp_mask]
    valid_grad_magnitudes = penguin_grad_magnitudes[valid_temp_mask]

    print(f"Filtering data for fitting: using temperatures >= {temp_threshold}°C")
    print(
        f"Data points for fitting: {len(valid_env_temps)} out of {len(penguin_env_temps)} total points"
    )

    temp_bins = np.linspace(np.min(valid_env_temps), np.max(valid_env_temps), 20)
    temp_centers = []
    avg_gradients = []

    for i in range(len(temp_bins) - 1):
        temp_mask = (valid_env_temps >= temp_bins[i]) & (
            valid_env_temps < temp_bins[i + 1]
        )
        if np.any(temp_mask):
            temp_center = (temp_bins[i] + temp_bins[i + 1]) / 2
            avg_grad = np.mean(valid_grad_magnitudes[temp_mask])
            temp_centers.append(temp_center)
            avg_gradients.append(avg_grad)

    temp_centers = np.array(temp_centers)
    avg_gradients = np.array(avg_gradients)

    # 高斯函數擬合
    try:
        # 估計初始參數
        max_grad = np.max(avg_gradients)
        min_grad = np.min(avg_gradients)
        peak_temp = temp_centers[np.argmax(avg_gradients)]
        temp_range = np.max(temp_centers) - np.min(temp_centers)

        # 初始猜測
        initial_guess = [
            max_grad - min_grad,  # A (amplitude)
            peak_temp,  # mu (peak position)
            temp_range / 4,  # sigma (width)
            min_grad,  # offset (baseline)
        ]

        popt_gauss, pcov_gauss = curve_fit(
            gaussian_func, temp_centers, avg_gradients, p0=initial_guess
        )
        A, mu, sigma, offset = popt_gauss

        # 計算 R² 值
        y_pred_gauss = gaussian_func(temp_centers, *popt_gauss)
        ss_res_gauss = np.sum((avg_gradients - y_pred_gauss) ** 2)
        ss_tot = np.sum((avg_gradients - np.mean(avg_gradients)) ** 2)
        r_squared_gauss = 1 - (ss_res_gauss / ss_tot)

        print("\n=== Gaussian Fit Results ===")
        print(
            f"Fitted function: y = {A:.3f} * exp(-((x - {mu:.3f})^2) / (2 * {sigma:.3f}^2)) + {offset:.3f}"
        )
        print(f"Peak position (mu): {mu:.3f} °C")
        print(f"Peak value: {A + offset:.3f} °C/m")
        print(f"Standard deviation (sigma): {sigma:.3f} °C")
        print(f"Baseline (offset): {offset:.3f} °C/m")
        print(f"R² = {r_squared_gauss:.4f}")

        # 計算參數的標準誤差
        param_errors_gauss = np.sqrt(np.diag(pcov_gauss))
        print(
            f"Parameter errors: A = ±{param_errors_gauss[0]:.3f}, mu = ±{param_errors_gauss[1]:.3f}, sigma = ±{param_errors_gauss[2]:.3f}, offset = ±{param_errors_gauss[3]:.3f}"
        )

    except Exception as e:
        print(f"Gaussian fitting failed: {e}")
        popt_gauss = None

        # 創建合併的圖表
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # 散點圖: 環境溫度 vs 梯度大小 (以體溫著色)
    scatter = ax.scatter(
        env_temps_down,
        grad_mag_down,
        alpha=0.3,
        s=6,
        c=body_temps_down,
        cmap="viridis",
        label="Individual Points",
    )

    # 溫度區間平均梯度
    ax.scatter(
        temp_centers,
        avg_gradients,
        color="red",
        s=80,
        alpha=0.9,
        edgecolors="black",
        linewidth=1,
        label="Binned Average",
        zorder=5,
    )

    # 繪製高斯擬合曲線
    temp_smooth = np.linspace(np.min(temp_centers), np.max(temp_centers), 200)

    if popt_gauss is not None:
        grad_smooth_gauss = gaussian_func(temp_smooth, *popt_gauss)
        ax.plot(
            temp_smooth,
            grad_smooth_gauss,
            "b-",
            linewidth=3,
            label=f"Gaussian fit: A={A:.2f}, μ={mu:.2f}°C, σ={sigma:.2f}°C, offset={offset:.2f}\nR² = {r_squared_gauss:.4f}",
            zorder=4,
        )

    ax.set_xlabel("Environment Temperature (°C)", fontsize=12)
    ax.set_ylabel("Gradient Magnitude (°C/m)", fontsize=12)
    ax.set_title(
        "Temperature Gradient Analysis with Gaussian Fit",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    # 添加顏色條
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Body Temperature (°C)", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    # 計算統計資訊
    print("\n=== Gradient Analysis Statistics ===")
    print(
        f"Environment temp range: [{np.min(penguin_env_temps):.3f}, {np.max(penguin_env_temps):.3f}] °C"
    )
    print(
        f"Body temp range: [{np.min(penguin_body_temps):.3f}, {np.max(penguin_body_temps):.3f}] °C"
    )
    print(
        f"Gradient magnitude range: [{np.min(penguin_grad_magnitudes):.3f}, {np.max(penguin_grad_magnitudes):.3f}] °C/m"
    )
    print(f"Gradient magnitude mean: {np.mean(penguin_grad_magnitudes):.3f} °C/m")
    print(f"Gradient magnitude std: {np.std(penguin_grad_magnitudes):.3f} °C/m")
    print(f"Gradient magnitude median: {np.median(penguin_grad_magnitudes):.3f} °C/m")

    # 分析不同溫度範圍的平均梯度 (較少區間)
    temp_bins_print = np.linspace(
        np.min(penguin_env_temps), np.max(penguin_env_temps), 10
    )
    print("\n=== Average Gradient by Temperature Range ===")
    for i in range(len(temp_bins_print) - 1):
        temp_mask = (penguin_env_temps >= temp_bins_print[i]) & (
            penguin_env_temps < temp_bins_print[i + 1]
        )
        if np.any(temp_mask):
            avg_grad = np.mean(penguin_grad_magnitudes[temp_mask])
            print(
                f"Temp range [{temp_bins_print[i]:.1f}, {temp_bins_print[i + 1]:.1f}): avg gradient = {avg_grad:.3f} °C/m (n={np.sum(temp_mask)})"
            )

    return (
        penguin_env_temps,
        penguin_grad_magnitudes,
        temp_centers,
        avg_gradients,
        popt_gauss if popt_gauss is not None else None,
    )


if __name__ == "__main__":
    main()
