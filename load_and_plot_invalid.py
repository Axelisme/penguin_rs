import matplotlib.pyplot as plt
import numpy as np

load_path = "penguin_simulation_data.npz"
save_path = load_path.replace(".npz", "_filtered_analysis.png")


def calculate_temperature_gradients(air_temps, box_size, num_grid):
    """Calculate temperature gradients for entire grid using numpy"""
    dx = box_size / num_grid
    dy = box_size / num_grid

    # Use numpy gradient to calculate gradients efficiently
    grad_y, grad_x = np.gradient(air_temps, dy, dx)

    # Calculate gradient magnitude
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    return grad_magnitude, grad_x, grad_y


prefer_temp = None


def calculate_filtered_data(
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
    PREFER_TEMP = params["PREFER_TEMP"]
    PENGUIN_MOVE_FACTOR = params["PENGUIN_MOVE_FACTOR"]

    NUM_FRAMES, NUM_PENGUINS = body_temps.shape
    dt = DT * STEPS_PER_FRAME

    global prefer_temp
    prefer_temp = PREFER_TEMP

    # Store filtered data points
    v_real_list = []
    grad_list = []
    v_predict_list = []
    env_temp_list = []
    body_temp_list = []

    # Find frame index for 50 seconds
    time_threshold = 50.0  # Filter out first 50 seconds
    start_frame_idx = np.searchsorted(times, time_threshold)

    # Process each frame
    for frame_idx in range(max(1, start_frame_idx), NUM_FRAMES):
        # Current and previous frame data
        curr_positions = positions[frame_idx]
        curr_body_temps = body_temps[frame_idx]
        curr_air_temps = air_temps[frame_idx]
        prev_air_temps = air_temps[frame_idx - 1]

        # Calculate temperature gradients for entire grid using numpy
        grad_magnitude, grad_x, grad_y = calculate_temperature_gradients(
            curr_air_temps, BOX_SIZE, NUM_GRID
        )

        # Convert positions to grid indices using vectorized operations
        grid_coords = np.clip(
            curr_positions / BOX_SIZE * NUM_GRID, 0, NUM_GRID - 1
        ).astype(int)

        # Get environment temperatures for all penguins at once
        curr_env_temps = curr_air_temps[grid_coords[:, 1], grid_coords[:, 0]]
        prev_env_temps = prev_air_temps[grid_coords[:, 1], grid_coords[:, 0]]

        # Apply filtering conditions using vectorized operations
        air_temp_mask = (-15 <= curr_env_temps) & (curr_env_temps <= 5)
        body_temp_mask = curr_body_temps <= 20
        valid_mask = air_temp_mask & body_temp_mask
        # valid_mask = np.ones_like(curr_body_temps, dtype=bool)

        # Calculate V_real for all valid penguins
        v_real_all = (curr_env_temps - prev_env_temps) / dt
        valid_v_real_mask = ~(np.isnan(v_real_all) | np.isinf(v_real_all))
        final_mask = valid_mask & valid_v_real_mask

        if not np.any(final_mask):
            continue

        # Get data for valid penguins only
        valid_indices = np.where(final_mask)[0]

        for penguin_idx in valid_indices:
            curr_body_temp = curr_body_temps[penguin_idx]
            curr_env_temp = curr_env_temps[penguin_idx]
            v_real = v_real_all[penguin_idx]

            grid_x, grid_y = grid_coords[penguin_idx]

            # Get temperature gradient at penguin position
            grad = grad_magnitude[grid_y, grid_x]

            # Calculate V_predict based on TTplot formula: -move_factor * (body_temp - prefer_temp) * gradient^2
            v_predict = (
                -PENGUIN_MOVE_FACTOR * (curr_body_temp - PREFER_TEMP) * (grad**2)
            )

            # Store data
            v_real_list.append(v_real)
            grad_list.append(grad)
            v_predict_list.append(v_predict)
            env_temp_list.append(curr_env_temp)
            body_temp_list.append(curr_body_temp)

    return (
        np.array(v_real_list),
        np.array(grad_list),
        np.array(v_predict_list),
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

    # Calculate filtered data
    (
        v_real_filtered,
        grad_filtered,
        v_predict_filtered,
        env_temp_filtered,
        body_temp_filtered,
    ) = calculate_filtered_data(positions, body_temps, air_temps, times, params)

    print(f"Filtered data points: {len(v_real_filtered)}")
    print(
        f"Air temp range: [{np.min(env_temp_filtered):.2f}, {np.max(env_temp_filtered):.2f}]"
    )
    print(
        f"Body temp range: [{np.min(body_temp_filtered):.2f}, {np.max(body_temp_filtered):.2f}]"
    )

    if len(v_real_filtered) == 0:
        print("No data points match the filtering criteria!")
        return

    # Downsample data for plotting efficiency
    downsample_factor = max(1, len(v_real_filtered) // 5000)  # Target ~5000 points
    downsample_indices = np.arange(0, len(v_real_filtered), downsample_factor)

    v_real_down = v_real_filtered[downsample_indices]
    grad_down = grad_filtered[downsample_indices]
    v_predict_down = v_predict_filtered[downsample_indices]
    env_temp_down = env_temp_filtered[downsample_indices]
    body_temp_down = body_temp_filtered[downsample_indices]

    # Calculate V_real - V_predict for y-axis
    # v_diff_down = v_real_down - v_predict_down

    # Create subplots (2x2 for 4 plots)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Calculate color range for V_predict (within two standard deviations)
    v_predict_mean = np.mean(v_predict_down)
    v_predict_std = np.std(v_predict_down)
    vmin = v_predict_mean - 2 * v_predict_std
    vmax = v_predict_mean + 2 * v_predict_std

    # Plot 1: (V_real - V_predict) vs Temperature Gradient (colored by V_predict)
    scatter1 = ax1.scatter(
        grad_down,
        v_real_down,
        alpha=0.6,
        s=8,
        c=v_predict_down,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    ax1.set_xlabel("Temperature Gradient (°C/m)")
    ax1.set_ylabel("V_real (°C/s)")
    ax1.set_title("V_real vs Temperature Gradient")
    ax1.grid(True, alpha=0.3)

    # Plot 2: V_real vs V_predict
    ax2.scatter(v_predict_down, v_real_down, alpha=0.6, s=8)
    ax2.set_xlabel("V_predict")
    ax2.set_ylabel("V_real")
    ax2.set_title("V_real vs V_predict")
    ax2.grid(True, alpha=0.3)

    # Plot 3: (V_real - V_predict) vs Environment Temperature (colored by V_predict)
    ax3.scatter(
        env_temp_down,
        v_real_down,
        alpha=0.6,
        s=8,
        c=v_predict_down,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    ax3.set_xlabel("Environment Temperature (°C)")
    ax3.set_ylabel("V_real (°C/s)")
    ax3.set_title("V_real vs Environment Temperature")
    ax3.grid(True, alpha=0.3)

    # Plot 4: V_real - V_predict vs (T_body - T_prefer) * Gradient
    # Calculate the new x-axis values for ax4
    # prefer_temp is a global variable set in calculate_filtered_data
    x_ax4_data = (body_temp_down - prefer_temp) * grad_down

    # Calculate the new y-axis values for ax4
    y_ax4_data = v_real_down - v_predict_down

    ax4.scatter(
        x_ax4_data,  # Use new x-axis data
        y_ax4_data,  # Use new y-axis data
        alpha=0.6,
        s=8,
        c=v_predict_down,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    ax4.set_xlabel("(T_body - T_prefer) * Gradient")  # Updated x-axis label
    ax4.set_ylabel("V_real - V_predict (°C/s)")  # Updated y-axis label
    ax4.set_title(
        "V_real - V_predict vs (T_body - T_prefer) * Gradient"
    )  # Updated title
    ax4.grid(True, alpha=0.3)

    # --- Robust cubic polynomial fit for ax4 ---

    # 1. Use full (non-downsampled) data for fitting
    x_full_for_ax4 = (body_temp_filtered - prefer_temp) * grad_filtered
    y_full_for_ax4 = v_real_filtered - v_predict_filtered

    # 2. Filter outliers using IQR method from the full data
    if len(y_full_for_ax4) >= 4:  # Need at least 4 points for robust IQR and cubic fit
        Q1 = np.percentile(y_full_for_ax4, 25)
        Q3 = np.percentile(y_full_for_ax4, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        non_outlier_mask = (y_full_for_ax4 >= lower_bound) & (
            y_full_for_ax4 <= upper_bound
        )
        x_fit_data = x_full_for_ax4[non_outlier_mask]
        y_fit_data = y_full_for_ax4[non_outlier_mask]

        if (
            len(x_fit_data) >= 4
        ):  # Check if enough points remain after outlier removal for cubic fit
            # Sort the data used for fitting
            sort_indices_fit = np.argsort(x_fit_data)
            x_fit_data_sorted = x_fit_data[sort_indices_fit]
            y_fit_data_sorted = y_fit_data[sort_indices_fit]

            # 3. Fit a cubic polynomial (ax^3 + cx + d) using the filtered, full-resolution data
            # Construct the design matrix for the specific cubic form ax^3 + cx + d
            # Columns are x^3, x, 1 (constant term)
            design_matrix = np.vstack(
                [
                    x_fit_data_sorted**3,
                    x_fit_data_sorted,  # Linear term
                    np.ones_like(x_fit_data_sorted),  # Constant term
                ]
            ).T

            # Solve for coefficients [a, c, d] using least squares
            coeffs, residuals, rank, singular_values = np.linalg.lstsq(
                design_matrix, y_fit_data_sorted, rcond=None
            )
            a_coeff, c_coeff, d_coeff = coeffs

            # Define the custom polynomial fit function
            def custom_poly_fit_func(x_vals, a, c, d):
                return a * x_vals**3 + c * x_vals + d

            # 4. Generate y-values for the fitted curve using the x-range of the plotted (downsampled) data
            sort_indices_plot = np.argsort(
                x_ax4_data
            )  # x_ax4_data is from downsampled data
            x_ax4_plot_sorted = x_ax4_data[sort_indices_plot]

            y_curve_on_plot = custom_poly_fit_func(
                x_ax4_plot_sorted, a_coeff, c_coeff, d_coeff
            )

            # Plot the fitted curve
            ax4.plot(
                x_ax4_plot_sorted,
                y_curve_on_plot,
                color="red",
                linestyle="--",
                label="Fit (ax^3+cx+d)",
            )
            ax4.set_ylim(y_ax4_data.min(), y_ax4_data.max())
            ax4.legend()

    else:
        print("Not enough data points for robust fitting.")
        if "poly_fit_func" in locals() or "poly_fit_func" in globals():
            ax4.legend().set_visible(False)

    # --- End of robust fit ---

    # Add color bar
    cbar = plt.colorbar(scatter1, ax=ax4, shrink=0.8)
    cbar.set_label("V_predict")

    # Add overall title
    fig.suptitle(
        f"Residual Analysis (V_real - V_predict): All Data\n"
        f"Data points: {len(v_real_down)}/{len(v_real_filtered)} (downsampled/total)",
        fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    # Calculate residuals for all data
    v_diff_all = v_real_filtered - v_predict_filtered

    # Print some statistics
    print("\nStatistics for all data:")
    print(
        f"V_real: mean={np.mean(v_real_filtered):.6f}, std={np.std(v_real_filtered):.6f}"
    )
    print(
        f"V_predict: mean={np.mean(v_predict_filtered):.6f}, std={np.std(v_predict_filtered):.6f}"
    )
    print(
        f"V_real - V_predict: mean={np.mean(v_diff_all):.6f}, std={np.std(v_diff_all):.6f}"
    )
    print(
        f"Gradient: mean={np.mean(grad_filtered):.6f}, std={np.std(grad_filtered):.6f}"
    )


if __name__ == "__main__":
    main()
