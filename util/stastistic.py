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
