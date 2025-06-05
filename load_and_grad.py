import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from util.stastistic import get_env_temps_at_positions, get_grad_at_positions

load_path = os.path.join("data", "N500_T100s_C(False)", "simulation.npz")
save_path = load_path.replace(".npz", "_grad.png")
# save_path = None


# def fit_func(x, A, mu, sigma, offset):
#     return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + offset


def fit_func(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e


def guess_initial_params(x, y):
    return (1, 1, 1, 1, 1)


def get_func_text(params, r_squared=None):
    return f"R² = {r_squared:.4f}"


def main():
    """
    Main function for temperature gradient analysis with user-defined fitting function

    Parameters:
    fit_func: user-defined function to use for fitting (required)
    """
    if fit_func is None:
        raise ValueError("fit_func must be provided by user")

    # Load npz file
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

    # Get parameters
    BOX_SIZE = params["BOX_SIZE"]
    NUM_FRAMES, NUM_PENGUINS = body_temps.shape

    # Find frame index corresponding to 50 seconds
    time_threshold = 50.0
    start_frame_idx = np.searchsorted(times, time_threshold)
    print(f"Filtering out first 50 seconds of data (frames 0-{start_frame_idx - 1})")
    print(f"Using frames {start_frame_idx}-{NUM_FRAMES - 1} for analysis")

    # Collect all data points
    penguin_positions_list = []
    penguin_env_temps_list = []
    penguin_body_temps_list = []
    penguin_grad_magnitudes_list = []
    penguin_grad_x_list = []
    penguin_grad_y_list = []

    print("Calculating gradients at penguin positions...")

    # Iterate through each time frame (starting from 50 seconds)
    for frame_idx in range(start_frame_idx, NUM_FRAMES):
        curr_positions = positions[frame_idx]
        curr_body_temps = body_temps[frame_idx]
        curr_air_temps = air_temps[frame_idx]

        # Get environment temperature at each penguin position
        env_temps_at_positions = get_env_temps_at_positions(
            curr_positions, curr_air_temps, BOX_SIZE
        )

        # Get temperature gradient at each penguin position
        gradients_at_positions = get_grad_at_positions(
            curr_positions, curr_air_temps, BOX_SIZE
        )

        # Calculate gradient magnitude
        grad_magnitudes = np.sqrt(
            gradients_at_positions[:, 0] ** 2 + gradients_at_positions[:, 1] ** 2
        )

        # Collect data
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

    # Convert to numpy arrays
    penguin_env_temps = np.array(penguin_env_temps_list)
    penguin_body_temps = np.array(penguin_body_temps_list)
    penguin_grad_magnitudes = np.array(penguin_grad_magnitudes_list)

    print(f"Total data points collected: {len(penguin_env_temps)}")

    # Downsample data for plotting efficiency
    downsample_factor = max(1, len(penguin_env_temps) // 8000)  # Target ~8000 points
    downsample_indices = np.arange(0, len(penguin_env_temps), downsample_factor)

    env_temps_down = penguin_env_temps[downsample_indices]
    body_temps_down = penguin_body_temps[downsample_indices]
    grad_mag_down = penguin_grad_magnitudes[downsample_indices]

    print(f"Downsampled to {len(env_temps_down)} points (factor: {downsample_factor})")

    # Calculate average gradient for temperature intervals (for fitting) - only use data above -15°C
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

    # Function fitting
    popt, pcov = curve_fit(
        fit_func,
        temp_centers,
        avg_gradients,
        p0=guess_initial_params(temp_centers, avg_gradients),
    )

    # Calculate R² value
    y_pred = fit_func(temp_centers, *popt)
    ss_res = np.sum((avg_gradients - y_pred) ** 2)
    ss_tot = np.sum((avg_gradients - np.mean(avg_gradients)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"Parameters: {popt}")

    # Create combined plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Scatter plot: environment temperature vs gradient magnitude (colored by body temperature)
    scatter = ax.scatter(
        env_temps_down, grad_mag_down, alpha=0.3, s=6, c=body_temps_down, cmap="viridis"
    )

    # Temperature interval average gradients
    ax.scatter(
        temp_centers,
        avg_gradients,
        color="red",
        s=80,
        alpha=0.9,
        edgecolors="black",
        linewidth=1,
        zorder=5,
    )

    # Plot fitting curve
    temp_smooth = np.linspace(np.min(temp_centers), np.max(temp_centers), 200)

    if popt is not None:
        grad_smooth = fit_func(temp_smooth, *popt)
        fit_label = f"R² = {r_squared:.4f}"
        ax.plot(temp_smooth, grad_smooth, "b-", linewidth=3, label=fit_label, zorder=4)

    ax.set_xlabel("Environment Temperature (°C)", fontsize=12)
    ax.set_ylabel("Gradient Magnitude (°C/m)", fontsize=12)
    ax.set_title(
        "Temperature Gradient Analysis with Function Fit",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Add color bar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Body Temperature (°C)", fontsize=11)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
