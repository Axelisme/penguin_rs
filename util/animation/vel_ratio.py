from typing import Any, List

import numpy as np
from matplotlib.axes import Axes
from sklearn.linear_model import HuberRegressor


class VelocityRatioPlot:
    def __init__(
        self,
        ax: Axes,
        body_temps: np.ndarray,
        env_temps: np.ndarray,
        gradients: np.ndarray,
        move_factor: float,
        prefer_temp: float,
        temp_room: float,
        dt: float,
    ) -> None:
        self.ax = ax
        self.move_factor = move_factor
        self.prefer_temp = prefer_temp
        self.temp_room = temp_room
        self.dt = dt

        self.prev_env_temps = env_temps

        # Calculate initial theoretical and actual velocities
        x_data, y_data, fit_slope = self._calculate_velocity_relationship(
            body_temps, env_temps, gradients
        )

        self.max_x_in_history = max(np.max(x_data), 1e-6)

        # Create scatter plot
        self.scatter = ax.scatter(
            x_data, y_data, alpha=0.6, s=8, c="blue", animated=True
        )

        # Create fit line (y = kx)
        x_range = np.array([np.nanmin(x_data), np.nanmax(x_data)])
        y_fit = fit_slope * x_range
        (self.fit_line,) = ax.plot(x_range, y_fit, "r-", linewidth=2, animated=True)

        y_limit = max(2 * np.abs(y_fit[-1]), 1e-6)
        self.ax.set_xlim(0.0, self.max_x_in_history)
        self.ax.set_ylim(-y_limit, y_limit)

        # Create text to display fit equation
        self.fit_text = ax.text(
            0.05,
            0.95,
            f"y = {fit_slope:.3f}x",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            verticalalignment="top",
        )
        # Set x and y axis labels
        self.ax.set_xlabel("V_predicted * (env_temp - temp_room)")
        self.ax.set_ylabel("V_actual - V_predicted")

    def _calculate_velocity_relationship(
        self,
        body_temps: np.ndarray,
        env_temps: np.ndarray,
        gradients: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Calculate the relationship data for plotting and fit slope"""

        v_reals = (env_temps - self.prev_env_temps) / self.dt
        v_predicts = -self.move_factor * (body_temps - self.prefer_temp) * gradients**2

        x_data = v_predicts * (env_temps - self.temp_room)
        y_data = v_reals - v_predicts

        # Calculate robust fit slope for y = kx (line through origin)
        # Using HuberRegressor to ignore outliers
        valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
        if np.sum(valid_mask) > 5:  # Need at least 5 points for robust fitting
            x_valid = x_data[valid_mask].reshape(-1, 1)
            y_valid = y_data[valid_mask]

            # normalize y and x to make them have normal scale
            scale = np.std(x_valid) + np.std(y_valid)
            x_valid = x_valid / scale
            y_valid = y_valid / scale

            # Use HuberRegressor with no intercept (force through origin)
            huber = HuberRegressor(fit_intercept=False, epsilon=1.35, max_iter=1000)
            huber.fit(x_valid, y_valid)
            fit_slope = huber.coef_[0]
        else:
            fit_slope = 0

        return x_data, y_data, fit_slope

    def update(
        self,
        body_temps: np.ndarray,
        env_temps: np.ndarray,
        gradients: np.ndarray,
    ) -> List[Any]:
        # Calculate new velocity relationship data
        x_data, y_data, fit_slope = self._calculate_velocity_relationship(
            body_temps, env_temps, gradients
        )

        self.max_x_in_history = max(np.max(x_data), self.max_x_in_history)
        self.prev_env_temps = env_temps

        # Update scatter plot data
        self.scatter.set_offsets(np.column_stack([x_data, y_data]))

        # Update fit line
        x_range = np.array([0.0, self.max_x_in_history])
        y_fit = fit_slope * x_range
        self.fit_line.set_data(x_range, y_fit)

        # Update text to display fit equation
        self.fit_text.set_text(f"y = {fit_slope:.3f}x")

        # Set axis limits
        # self.ax.set_xlim(0.0, 100)
        # self.ax.set_ylim(-6, 6)
        y_limit = max(2 * np.abs(y_fit[-1]), 1e-6)
        self.ax.set_xlim(0.0, self.max_x_in_history)
        self.ax.set_ylim(-y_limit, y_limit)

        return [self.scatter, self.fit_line, self.fit_text]
