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

        # Create scatter plot
        self.scatter = ax.scatter(
            x_data, y_data, alpha=0.6, s=8, c="blue", animated=True
        )

        # Create fit line (y = kx)
        x_range = np.array([np.nanmin(x_data), np.nanmax(x_data)])
        y_fit = fit_slope * x_range
        (self.fit_line,) = ax.plot(x_range, y_fit, "r-", linewidth=2, animated=True)

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
        # self.ax.set_xlabel("V_predicted * (env_temp - temp_room)")
        self.ax.set_xlabel("V_predicted * (body_temp - prefer_temp)")
        self.ax.set_ylabel("V_actual - V_predicted")

        self.std_y_in_history = [1e-6]
        self.std_x_in_history = [1e-6]

        self._update_axis_limits(x_data, y_data)

    def _update_axis_limits(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        self.std_x_in_history.append(np.nanstd(x_data))
        self.std_y_in_history.append(np.nanstd(y_data))

        # Set axis limits
        x_limit = np.nanmean(self.std_x_in_history) * 8
        y_limit = np.nanmean(self.std_y_in_history) * 8
        self.ax.set_xlim(-x_limit, x_limit)
        self.ax.set_ylim(-y_limit, y_limit)
        # self.ax.set_xlim(0.0, 100)
        # self.ax.set_ylim(-6, 6)

    def _calculate_velocity_relationship(
        self,
        body_temps: np.ndarray,
        env_temps: np.ndarray,
        gradients: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Calculate the relationship data for plotting and fit slope"""

        v_reals = (env_temps - self.prev_env_temps) / self.dt
        v_predicts = -self.move_factor * (body_temps - self.prefer_temp) * gradients**2

        # x_data = v_predicts * (env_temps - 5.0)
        x_data = v_predicts * (body_temps - self.prefer_temp)
        # x_data = v_predicts
        y_data = v_reals - v_predicts
        # y_data = v_reals

        # Calculate robust fit slope for y = kx (line through origin)
        # Using HuberRegressor to ignore outliers
        valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
        if np.sum(valid_mask) > 5:  # Need at least 5 points for robust fitting
            x_valid = x_data[valid_mask].reshape(-1, 1)
            y_valid = y_data[valid_mask]

            # normalize y and x to make them have normal scale
            scale = np.nanstd(x_valid) + np.nanstd(y_valid)
            if scale > 0:
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

        # update the max x in history with a weighted average
        self.prev_env_temps = env_temps

        # Update scatter plot data
        self.scatter.set_offsets(np.column_stack([x_data, y_data]))
        self.scatter.set_color(np.where(body_temps > self.prefer_temp, "red", "blue"))

        # Update fit line
        x_limit = np.nanmean(self.std_x_in_history) * 10
        x_range = np.array([-x_limit, x_limit])
        y_fit = fit_slope * x_range
        self.fit_line.set_data(x_range, y_fit)

        # Update text to display fit equation
        self.fit_text.set_text(f"y = {fit_slope:+.3f}x")

        self._update_axis_limits(x_data, y_data)

        return [self.scatter, self.fit_line, self.fit_text]


class VelocityRatioPlot2:
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
        x_data, y_data, red_mask, fit_slope_red, fit_slope_blue = (
            self._calculate_velocity_relationship(body_temps, env_temps, gradients)
        )

        # Create scatter plot
        self.scatter = ax.scatter(
            x_data, y_data, alpha=0.6, s=8, c="blue", animated=True
        )

        # Create fit lines for red and blue points (y = kx)
        x_range = np.array([np.nanmin(x_data), np.nanmax(x_data)])
        y_fit_red = fit_slope_red * x_range
        y_fit_blue = fit_slope_blue * x_range
        (self.fit_line_red,) = ax.plot(
            x_range,
            y_fit_red,
            "r-",
            linewidth=2,
            animated=True,
            label="Hot (T > prefer)",
        )
        (self.fit_line_blue,) = ax.plot(
            x_range,
            y_fit_blue,
            "b-",
            linewidth=2,
            animated=True,
            label="Cold (T < prefer)",
        )

        # Create text to display fit equations
        self.fit_text_red = ax.text(
            0.05,
            0.95,
            f"Red: y = {fit_slope_red:.3f}x",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3),
            verticalalignment="top",
        )

        self.fit_text_blue = ax.text(
            0.05,
            0.85,
            f"Blue: y = {fit_slope_blue:.3f}x",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="blue", alpha=0.3),
            verticalalignment="top",
        )

        # Set x and y axis labels
        # self.ax.set_xlabel("V_predicted * (env_temp - temp_room)")
        self.ax.set_xlabel("V_predicted * (body_temp - prefer_temp)")
        self.ax.set_ylabel("V_actual - V_predicted")

        self.std_y_in_history = [1e-6]
        self.std_x_in_history = [1e-6]

        self._update_axis_limits(x_data, y_data)

    def _update_axis_limits(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        self.std_x_in_history.append(np.nanstd(x_data))
        self.std_y_in_history.append(np.nanstd(y_data))

        # Set axis limits
        x_limit = np.nanmean(self.std_x_in_history) * 8
        y_limit = np.nanmean(self.std_y_in_history) * 8
        self.ax.set_xlim(-x_limit, 0.0)
        self.ax.set_ylim(-y_limit, y_limit)
        # self.ax.set_xlim(0.0, 100)
        # self.ax.set_ylim(-6, 6)

    def _calculate_velocity_relationship(
        self,
        body_temps: np.ndarray,
        env_temps: np.ndarray,
        gradients: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Calculate the relationship data for plotting and fit slopes for red and blue points"""

        v_reals = (env_temps - self.prev_env_temps) / self.dt
        v_predicts = -self.move_factor * (body_temps - self.prefer_temp) * gradients**2

        # x_data = v_predicts * (env_temps - 5.0)
        x_data = v_predicts * (body_temps - self.prefer_temp)
        y_data = v_reals - v_predicts

        # Create mask for red points (body_temp > prefer_temp)
        red_mask = body_temps > self.prefer_temp
        blue_mask = ~red_mask

        # Calculate robust fit slope for red points (y = kx, line through origin)
        fit_slope_red = self._fit_points(x_data[red_mask], y_data[red_mask])

        # Calculate robust fit slope for blue points (y = kx, line through origin)
        fit_slope_blue = self._fit_points(x_data[blue_mask], y_data[blue_mask])

        return x_data, y_data, red_mask, fit_slope_red, fit_slope_blue

    def _fit_points(self, x_data: np.ndarray, y_data: np.ndarray) -> float:
        """Fit points using HuberRegressor and return slope"""
        valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
        if np.sum(valid_mask) > 5:  # Need at least 5 points for robust fitting
            x_valid = x_data[valid_mask].reshape(-1, 1)
            y_valid = y_data[valid_mask]

            # normalize y and x to make them have normal scale
            scale = np.nanstd(x_valid) + np.nanstd(y_valid)
            if scale > 0:
                x_valid = x_valid / scale
                y_valid = y_valid / scale

            # Use HuberRegressor with no intercept (force through origin)
            huber = HuberRegressor(fit_intercept=False, epsilon=1.35, max_iter=1000)
            huber.fit(x_valid, y_valid)
            return huber.coef_[0]
        else:
            return 0

    def update(
        self,
        body_temps: np.ndarray,
        env_temps: np.ndarray,
        gradients: np.ndarray,
    ) -> List[Any]:
        # Calculate new velocity relationship data
        x_data, y_data, red_mask, fit_slope_red, fit_slope_blue = (
            self._calculate_velocity_relationship(body_temps, env_temps, gradients)
        )

        # update the max x in history with a weighted average
        self.prev_env_temps = env_temps

        # Update scatter plot data
        self.scatter.set_offsets(np.column_stack([x_data, y_data]))
        self.scatter.set_color(np.where(red_mask, "red", "blue"))

        # Update fit lines
        x_limit = np.nanmean(self.std_x_in_history) * 10
        x_range = np.array([-x_limit, x_limit])

        # Update red fit line
        y_fit_red = fit_slope_red * x_range
        self.fit_line_red.set_data(x_range, y_fit_red)

        # Update blue fit line
        y_fit_blue = fit_slope_blue * x_range
        self.fit_line_blue.set_data(x_range, y_fit_blue)

        # Update text to display fit equations
        self.fit_text_red.set_text(f"Red: y = {fit_slope_red:+.3f}x")
        self.fit_text_blue.set_text(f"Blue: y = {fit_slope_blue:+.3f}x")

        self._update_axis_limits(x_data, y_data)

        return [
            self.scatter,
            self.fit_line_red,
            self.fit_line_blue,
            self.fit_text_red,
            self.fit_text_blue,
        ]
