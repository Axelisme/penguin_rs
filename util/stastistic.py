import numpy as np


def calculate_temperature_gradient(air_temp_grid, box_size):
    """Calculate spatial temperature gradient magnitude with correct grid spacing"""
    num_grid = air_temp_grid.shape[0]
    grid_spacing = box_size / num_grid

    # Calculate gradients using numpy gradient with correct spacing
    grad_x, grad_y = np.gradient(air_temp_grid, grid_spacing)
    return grad_x, grad_y


def calculate_temp_gradient_relationship(
    air_temp_grid, box_size, smooth_range=0.05, num_bins=50
):
    """Calculate relationship between temperature and average gradient with smoothing"""
    grad_x, grad_y = calculate_temperature_gradient(air_temp_grid, box_size)

    # Get temperature range
    temp_min, temp_max = np.min(air_temp_grid), np.max(air_temp_grid)
    temp_range = np.linspace(temp_min, temp_max, num_bins)

    gradients = []

    for target_temp in temp_range:
        # Create mask for temperatures within smooth_range of target_temp
        temp_mask = np.abs(air_temp_grid - target_temp) <= smooth_range

        if np.any(temp_mask):
            # Calculate average gradient for this temperature range
            grad_amp = np.sqrt(grad_x[temp_mask] ** 2 + grad_y[temp_mask] ** 2)
            avg_gradient = np.mean(grad_amp)
        else:
            avg_gradient = 0.0

        gradients.append(avg_gradient)

    return temp_range, np.array(gradients)


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


def get_grad_at_positions(positions, air_temp_grid, box_size):
    """Get temperature gradients at each penguin position using bilinear interpolation"""
    num_grid = air_temp_grid.shape[0]
    gradients = []

    # Pre-compute gradients at all grid points
    grad_x, grad_y = calculate_temperature_gradient(air_temp_grid, box_size)

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

        # Bilinear interpolation for gradients
        interp_grad_x = (
            grad_x[x0, y0] * (1 - fx) * (1 - fy)
            + grad_x[x1, y0] * fx * (1 - fy)
            + grad_x[x0, y1] * (1 - fx) * fy
            + grad_x[x1, y1] * fx * fy
        )

        interp_grad_y = (
            grad_y[x0, y0] * (1 - fx) * (1 - fy)
            + grad_y[x1, y0] * fx * (1 - fy)
            + grad_y[x0, y1] * (1 - fx) * fy
            + grad_y[x1, y1] * fx * fy
        )

        gradients.append([interp_grad_x, interp_grad_y])

    return np.array(gradients)


def analyze_check_env_deviations(
    velocities, env_temps, gradients, dt, air_temp_grid, box_size
):
    """
    Analyze and explain deviations in the V·∇T vs dT_env/dt relationship

    Returns a dictionary with analysis results explaining why the scatter plot
    has standard deviation from the ideal y=x line.
    """

    # Calculate basic relationship
    dT_dt_actual = (
        np.gradient(env_temps) / dt if len(env_temps) > 1 else np.zeros_like(env_temps)
    )
    v_dot_grad = -np.sum(velocities * gradients, axis=1)

    # Calculate deviations
    residuals = dT_dt_actual - v_dot_grad
    std_dev = np.std(residuals)
    correlation = (
        np.corrcoef(v_dot_grad, dT_dt_actual)[0, 1] if len(v_dot_grad) > 1 else 0
    )

    # Estimate contributions to deviation
    analysis = {
        "correlation": correlation,
        "std_deviation": std_dev,
        "num_points": len(residuals),
        "explanations": [],
    }

    # 1. Check if time step is too large for numerical accuracy
    typical_velocity = np.mean(np.linalg.norm(velocities, axis=1))
    typical_gradient = np.mean(np.linalg.norm(gradients, axis=1))
    if dt > 0.05:  # Arbitrary threshold for "large" time step
        analysis["explanations"].append(
            f"Large time step (dt={dt:.3f}s) causes numerical errors in dT/dt calculation"
        )

    # 2. Check for diffusion effects
    diffusion_estimate = 0.4 * np.mean(np.abs(np.gradient(np.gradient(air_temp_grid))))
    advection_estimate = typical_velocity * typical_gradient
    if diffusion_estimate > 0.1 * advection_estimate:
        analysis["explanations"].append(
            f"Diffusion effects (D∇²T) contribute {diffusion_estimate / advection_estimate * 100:.1f}% compared to advection"
        )

    # 3. Check for outliers
    outlier_fraction = np.sum(np.abs(residuals) > 3 * std_dev) / len(residuals)
    if outlier_fraction > 0.05:
        analysis["explanations"].append(
            f"Outliers present: {outlier_fraction * 100:.1f}% of points are >3σ from mean"
        )

    # 4. Check spatial resolution effects
    num_grid = air_temp_grid.shape[0]
    grid_spacing = box_size / num_grid
    if grid_spacing > 0.1:  # Arbitrary threshold
        analysis["explanations"].append(
            f"Coarse spatial grid (spacing={grid_spacing:.3f}) may cause interpolation errors"
        )

    return analysis
