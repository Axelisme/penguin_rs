import os

import matplotlib.pyplot as plt
import numpy as np

load_path = os.path.join("data", "N500_T100s_C(False)", "simulation.npz")
save_path = load_path.replace(".npz", "_phase2d.png")
# save_path = None


def calculate_velocity_data_all_time(
    positions: np.ndarray,
    body_temps: np.ndarray,
    air_temps: np.ndarray,
    times: np.ndarray,
    params: dict,
):
    DT = params["DT"]
    STEPS_PER_FRAME = params["STEPS_PER_FRAME"]
    BOX_SIZE = params["BOX_SIZE"]
    NUM_GRID = params["NUM_GRID"]

    NUM_FRAMES, NUM_PENGUINS = body_temps.shape
    dt = DT * STEPS_PER_FRAME

    # Store data points needed for plotting
    y_data_list = []
    env_temp_list = []
    body_temp_list = []

    # Find frame index for 50 seconds
    time_threshold = 50.0  # Filter out first 50 seconds
    start_frame_idx = np.searchsorted(times, time_threshold)

    # Calculate environment temperature for each penguin at each time point
    for frame_idx in range(max(1, start_frame_idx), NUM_FRAMES):
        # Current and previous frame data
        curr_positions = positions[frame_idx]
        curr_body_temps = body_temps[frame_idx]
        curr_air_temps = air_temps[frame_idx]
        prev_air_temps = air_temps[frame_idx - 1]

        for penguin_idx in range(NUM_PENGUINS):
            # Calculate environment temperature at penguin position
            curr_pos = curr_positions[penguin_idx]

            # Convert position to grid indices
            curr_grid_x = int(
                np.clip(curr_pos[0] / BOX_SIZE * NUM_GRID, 0, NUM_GRID - 1)
            )
            curr_grid_y = int(
                np.clip(curr_pos[1] / BOX_SIZE * NUM_GRID, 0, NUM_GRID - 1)
            )

            # Get environment temperatures
            curr_env_temp = curr_air_temps[curr_grid_y, curr_grid_x]
            prev_env_temp = prev_air_temps[curr_grid_y, curr_grid_x]

            # Calculate actual velocity (environment temperature change rate)
            v_real = (curr_env_temp - prev_env_temp) / dt

            curr_body_temp = curr_body_temps[penguin_idx]

            # Keep only valid data points
            if not (np.isnan(v_real) or np.isinf(v_real)):
                y_data_list.append(v_real)
                env_temp_list.append(curr_env_temp)
                body_temp_list.append(curr_body_temp)

    return (
        np.array(y_data_list),
        np.array(env_temp_list),
        np.array(body_temp_list),
    )


def main():
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

    # Calculate velocity data for all time points
    y_data_all, env_temp_all, body_temp_all = calculate_velocity_data_all_time(
        positions, body_temps, air_temps, times, params
    )

    print(f"Total data points: {len(y_data_all)}")

    # Downsample data for plotting efficiency
    downsample_factor = max(1, len(y_data_all) // 5000)  # Target ~5000 points
    downsample_indices = np.arange(0, len(y_data_all), downsample_factor)

    y_downsampled = y_data_all[downsample_indices]
    env_temp_downsampled = env_temp_all[downsample_indices]
    body_temp_downsampled = body_temp_all[downsample_indices]

    # Create 2D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)

    # Calculate color range (within two standard deviations)
    v_real_mean = np.mean(y_downsampled)
    v_real_std = np.std(y_downsampled)
    vmin = v_real_mean - 4 * v_real_std
    vmax = v_real_mean + 4 * v_real_std

    # Scatter plot: x=body_temp, y=env_temp, color=V_real
    scatter = ax.scatter(
        body_temp_downsampled,
        env_temp_downsampled,
        alpha=0.6,
        s=8,
        c=y_downsampled,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        label=f"Data points (n={len(y_downsampled)})",
    )

    # Add color bar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("V_real")

    ax.set_xlim(17, 23)
    ax.set_ylim(-30, 15)

    ax.set_xlabel("Body Temperature (°C)")
    ax.set_ylabel("Environment Temperature (°C)")
    ax.set_title(
        "2D Scatter: Environment Temperature vs Body Temperature (colored by V_real)"
    )

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()
