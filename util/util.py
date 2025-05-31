from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def calculate_temperature_gradient(air_temp_grid):
    """Calculate spatial temperature gradient magnitude"""
    # Calculate gradients using numpy gradient
    grad_x, grad_y = np.gradient(air_temp_grid)
    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return gradient_magnitude


def calculate_temp_gradient_relationship(air_temp_grid, smooth_range=0.1, num_bins=50):
    """Calculate relationship between temperature and average gradient with smoothing"""
    gradient_magnitude = calculate_temperature_gradient(air_temp_grid)

    # Get temperature range
    temp_min, temp_max = np.min(air_temp_grid), np.max(air_temp_grid)
    temp_range = np.linspace(temp_min, temp_max, num_bins)

    avg_gradients = []

    for target_temp in temp_range:
        # Create mask for temperatures within smooth_range of target_temp
        temp_mask = np.abs(air_temp_grid - target_temp) <= smooth_range

        if np.any(temp_mask):
            # Calculate average gradient for this temperature range
            avg_gradient = np.mean(gradient_magnitude[temp_mask])
        else:
            avg_gradient = 0.0

        avg_gradients.append(avg_gradient)

    return temp_range, np.array(avg_gradients)


def calculate_penguin_density_by_temp(
    positions, air_temp_grid, box_size, num_grid, temp_range=None, num_bins=50
):
    """Calculate penguin density at different temperature ranges"""
    if temp_range is None:
        temp_min, temp_max = np.min(air_temp_grid), np.max(air_temp_grid)
        temp_range = np.linspace(temp_min, temp_max, num_bins)

    densities = []
    grid_cell_area = (box_size / num_grid) ** 2  # Area of each grid cell

    for target_temp in temp_range:
        # Create mask for grid cells within temperature range (±0.5°C)
        temp_tolerance = 1.0
        temp_mask = np.abs(air_temp_grid - target_temp) <= temp_tolerance

        if np.any(temp_mask):
            # Count penguins in these temperature zones
            penguin_count = 0
            total_area = np.sum(temp_mask) * grid_cell_area

            for pos in positions:
                # Convert position to grid indices
                i = int(np.clip(pos[0] / box_size * num_grid, 0, num_grid - 1))
                j = int(np.clip(pos[1] / box_size * num_grid, 0, num_grid - 1))

                if temp_mask[i, j]:
                    penguin_count += 1

            # Calculate density (penguins per unit area)
            density = penguin_count / total_area if total_area > 0 else 0.0
        else:
            density = 0.0

        densities.append(density)

    return temp_range, np.array(densities)


def get_env_temps_at_positions(positions, air_temp_grid, box_size, num_grid):
    """Get environmental temperature at each penguin position"""
    env_temps = []
    for pos in positions:
        # Convert position to grid indices
        i = int(np.clip(pos[0] / box_size * num_grid + 0.5, 0, num_grid - 1))
        j = int(np.clip(pos[1] / box_size * num_grid + 0.5, 0, num_grid - 1))
        temp = np.mean(air_temp_grid[i - 1 : i + 2, j - 1 : j + 2])
        env_temps.append(temp)
    return np.array(env_temps)


def create_penguin_colors(body_temps, prefer_temp_common):
    """Create binary colors based on temperature preference"""
    return ["white" if temp > prefer_temp_common else "black" for temp in body_temps]


def calculate_temperature_stats(air_temp_grid, body_temps):
    """Calculate temperature statistics for display"""
    air_min, air_mid, air_max = (
        np.min(air_temp_grid),
        np.median(air_temp_grid),
        np.max(air_temp_grid),
    )
    body_min, body_mid, body_max = (
        np.min(body_temps),
        np.median(body_temps),
        np.max(body_temps),
    )
    return (air_min, air_mid, air_max), (body_min, body_mid, body_max)


def update_axis_limits(ax, data_range, padding_factor=0.1):
    """Update axis limits with padding"""
    data_min, data_max = np.min(data_range), np.max(data_range)
    if data_min != data_max:
        data_pad = (data_max - data_min) * padding_factor
        ax.set_ylim(data_min - data_pad, data_max + data_pad)
    else:
        ax.set_ylim(0, max(1, data_max * 1.1))


def update_color_limits(im, data, padding_factor=0.1):
    """Update image color limits with padding"""
    data_min, data_max = np.min(data), np.max(data)
    if data_min == data_max:
        data_min -= 0.5
        data_max += 0.5
    im_pad = (data_max - data_min) * padding_factor
    im.set_clim(vmin=data_min - im_pad, vmax=data_max + im_pad)


def create_title_text(current_sim_time, frame, total_frames, air_stats, body_stats):
    """Create formatted title text for animation"""
    air_min, air_mid, air_max = air_stats
    body_min, body_mid, body_max = body_stats

    return (
        f"Penguin Sim - Time: {current_sim_time:.2f}s Frame: {frame}/{total_frames} "
        f"Body T: [{body_min:.2f}, {body_mid:.2f},{body_max:.2f}] "
        f"Air T: [{air_min:.2f}, {air_mid:.2f},{air_max:.2f}]"
    )


def plot_integrated_analysis(
    A: float,
    B: float,
    x0: float,
    H_func: Callable,
    trajectories: List[np.ndarray],
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    grid_density: int = 20,
) -> None:
    """
    Plot integrated analysis: vector field + trajectories + isoclines in one figure

    Args:
        A: parameter A
        B: parameter B
        x0: parameter x0 (reference point for dy/dt equation)
        H_func: function H(x, y)
        trajectories: list of trajectory data
        x_range: x-axis range
        y_range: y-axis range
        grid_density: grid density for vector field
    """
    plt.figure(figsize=(14, 10))

    # Create grid for vector field
    x = np.linspace(x_range[0], x_range[1], grid_density)
    y = np.linspace(y_range[0], y_range[1], grid_density)
    X, Y = np.meshgrid(x, y)

    # Calculate vector field
    DX = A - B * (X - Y)
    DY = (X - x0) * H_func(X, Y)

    # Normalize vector length for better display
    M = np.sqrt(DX**2 + DY**2)
    M[M == 0] = 1  # Avoid division by zero
    DX_norm = DX / M
    DY_norm = DY / M

    # Plot vector field
    quiver = plt.quiver(X, Y, DX_norm, DY_norm, M, cmap="viridis", alpha=0.6)
    plt.colorbar(quiver, label="Vector magnitude", shrink=0.8)

    # Add zero isoclines
    plt.contour(
        X, Y, DX, levels=[0], colors="red", linestyles="--", alpha=0.8, linewidths=2
    )
    plt.contour(
        X, Y, DY, levels=[0], colors="blue", linestyles="--", alpha=0.8, linewidths=2
    )

    # Plot trajectories
    for trajectory in trajectories:
        plt.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.9, linewidth=0.5)

        # plot end point
        plt.plot(trajectory[-1, 0], trajectory[-1, 1], "o", color="red", markersize=5)

    # Customize plot
    plt.xlim(x_range[0], x_range[1])
    plt.ylim(y_range[0], y_range[1])
    plt.xlabel("x", fontsize=14)
    plt.ylabel("y", fontsize=14)
    plt.title(f"dx/dt = {A} - {B}(x-y), dy/dt = (x-{x0})H(x,y)\n", fontsize=16, pad=20)

    plt.tight_layout()
    plt.show()
