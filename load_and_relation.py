import os

import matplotlib.pyplot as plt
import numpy as np

from util.stastistic import get_env_temps_at_positions, get_grad_at_positions

load_path = os.path.join("data", "N500_T100s_C(True)", "simulation.npz")
save_path = load_path.replace(".npz", "_four_plots.png")
# save_path = None

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


# def grad_func(y):
#     a = -5.02714357e-04
#     b = -1.76880183e-02
#     c = -2.31417227e-01
#     d = -1.83132979e-01
#     e = 2.77934649e01
#     grad = a * y**4 + b * y**3 + c * y**2 + d * y + e
#     return np.clip(grad, 0.0, None)


def calculate_predicted_env_temp_change(body_temps, env_temps):
    grad_values = grad_func(env_temps)
    predicted_dydt = (
        -PENGUIN_MOVE_FACTOR * (body_temps - PREFER_TEMP) * (grad_values**2)
    )
    return predicted_dydt


def main():
    # Load simulation data
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

    # Collect data from frames after 50s
    actual_env_temp_changes = []
    predicted_env_temp_changes = []
    env_temps_list = []
    body_temps_list = []
    grad_magnitudes_list = []

    print("Processing data from 50s onwards...")

    # Calculate dt from time array
    dt = np.mean(np.diff(times))
    print(f"Time step dt = {dt:.4f} seconds")

    # Process frames from 50s onwards
    for frame_idx in range(start_frame_idx, NUM_FRAMES - 1):  # -1 to avoid index error
        curr_positions = positions[frame_idx]
        next_positions = positions[frame_idx + 1]
        curr_body_temps = body_temps[frame_idx]
        curr_air_temps = air_temps[frame_idx]
        next_air_temps = air_temps[frame_idx + 1]

        # Get environmental temperatures at penguin positions
        curr_env_temps = get_env_temps_at_positions(
            curr_positions, curr_air_temps, BOX_SIZE
        )
        next_env_temps = get_env_temps_at_positions(
            next_positions, next_air_temps, BOX_SIZE
        )

        # Calculate actual environmental temperature change
        actual_dT_dt = (next_env_temps - curr_env_temps) / dt

        # Calculate predicted environmental temperature change
        predicted_dT_dt = calculate_predicted_env_temp_change(
            curr_body_temps, curr_env_temps
        )

        # Get temperature gradients
        gradients = get_grad_at_positions(curr_positions, curr_air_temps, BOX_SIZE)
        grad_magnitudes = np.sqrt(gradients[:, 0] ** 2 + gradients[:, 1] ** 2)

        # Collect valid data points (remove NaN and inf values)
        for penguin_idx in range(NUM_PENGUINS):
            if (
                not np.isnan(actual_dT_dt[penguin_idx])
                and not np.isinf(actual_dT_dt[penguin_idx])
                and not np.isnan(predicted_dT_dt[penguin_idx])
                and not np.isinf(predicted_dT_dt[penguin_idx])
                and not np.isnan(grad_magnitudes[penguin_idx])
                and not np.isinf(grad_magnitudes[penguin_idx])
            ):
                actual_env_temp_changes.append(actual_dT_dt[penguin_idx])
                predicted_env_temp_changes.append(predicted_dT_dt[penguin_idx])
                env_temps_list.append(curr_env_temps[penguin_idx])
                body_temps_list.append(curr_body_temps[penguin_idx])
                grad_magnitudes_list.append(grad_magnitudes[penguin_idx])

    # Convert to numpy arrays
    actual_changes = np.array(actual_env_temp_changes)
    predicted_changes = np.array(predicted_env_temp_changes)
    env_temps = np.array(env_temps_list)
    body_temps = np.array(body_temps_list)
    grad_mags = np.array(grad_magnitudes_list)

    print(f"Total valid data points collected: {len(actual_changes)}")

    # Downsample for plotting efficiency
    downsample_factor = max(1, len(actual_changes) // 5000)  # Target ~5000 points
    downsample_indices = np.arange(0, len(actual_changes), downsample_factor)

    actual_down = actual_changes[downsample_indices]
    predicted_down = predicted_changes[downsample_indices]
    env_temps_down = env_temps[downsample_indices]
    body_temps_down = body_temps[downsample_indices]
    grad_mags_down = grad_mags[downsample_indices]

    print(f"Downsampled to {len(actual_down)} points (factor: {downsample_factor})")

    # Calculate correlation coefficient for actual vs predicted
    correlation = np.corrcoef(actual_down, predicted_down)[0, 1]
    print(f"Correlation between actual and predicted changes: {correlation:.4f}")

    # Create four subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Top-left: Actual vs Predicted environmental temperature change
    ax1.scatter(predicted_down, actual_down, alpha=0.5, s=8, c="blue")

    y_limit = 5 * max(np.std(actual_down), np.std(predicted_down))
    ax1.set_ylim(-y_limit, y_limit)
    ax1.set_xlim(-y_limit, y_limit)
    ax1.set_xlabel("Predicted dT_env/dt (°C/s)")
    ax1.set_ylabel("Actual dT_env/dt (°C/s)")
    ax1.set_title(
        f"Actual vs Predicted Environmental Temperature Change\nCorrelation: {correlation:.4f}"
    )
    ax1.grid(True, alpha=0.3)

    # Top-right: Actual environmental temperature change vs Environmental temperature
    scatter2 = ax2.scatter(env_temps_down, actual_down, alpha=0.5, s=8)
    ax2.set_xlabel("Environmental Temperature (°C)")
    ax2.set_ylabel("Actual dT_env/dt (°C/s)")
    ax2.set_title("Environmental Temperature Change vs Environment Temperature")
    ax2.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label("Predicted dT_env/dt (°C/s)")

    # Bottom-left: Actual environmental temperature change vs Body temperature
    scatter3 = ax3.scatter(body_temps_down, actual_down, alpha=0.5, s=8)
    ax3.set_xlabel("Body Temperature (°C)")
    ax3.set_ylabel("Actual dT_env/dt (°C/s)")
    ax3.set_title("Environmental Temperature Change vs Body Temperature")
    ax3.grid(True, alpha=0.3)
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_label("Predicted dT_env/dt (°C/s)")

    # Bottom-right: Actual environmental temperature change vs Temperature gradient magnitude
    scatter4 = ax4.scatter(grad_mags_down, actual_down, alpha=0.5, s=8)
    ax4.set_xlabel("Temperature Gradient Magnitude (°C/m)")
    ax4.set_ylabel("Actual dT_env/dt (°C/s)")
    ax4.set_title("Environmental Temperature Change vs Temperature Gradient")
    ax4.grid(True, alpha=0.3)
    cbar4 = plt.colorbar(scatter4, ax=ax4)
    cbar4.set_label("Predicted dT_env/dt (°C/s)")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    main()
