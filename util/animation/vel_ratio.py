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
        self.ax.set_xlabel("V_predicted * (body_temp - temp_room)")
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
        y_data = v_reals - v_predicts

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

        # Update fit line
        x_limit = np.nanmean(self.std_x_in_history) * 10
        x_range = np.array([-x_limit, x_limit])
        y_fit = fit_slope * x_range
        self.fit_line.set_data(x_range, y_fit)

        # Update text to display fit equation
        self.fit_text.set_text(f"y = {fit_slope:.3f}x")

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
        x_data, y_data, fit_slope_pos, fit_slope_neg = (
            self._calculate_velocity_relationship(body_temps, env_temps, gradients)
        )

        # Create scatter plot
        self.scatter = ax.scatter(
            x_data, y_data, alpha=0.6, s=8, c="green", animated=True
        )

        # Positive x fit line (blue)
        x_pos_range = (
            np.array([0, np.nanmax(x_data)])
            if np.nanmax(x_data) > 0
            else np.array([0, 1])
        )
        y_pos_fit = fit_slope_pos * x_pos_range
        (self.fit_line_pos,) = ax.plot(
            x_pos_range,
            y_pos_fit,
            "b-",
            linewidth=2,
            animated=True,
            label="Positive x fit",
        )

        # Negative x fit line (red)
        x_neg_range = (
            np.array([np.nanmin(x_data), 0])
            if np.nanmin(x_data) < 0
            else np.array([-1, 0])
        )
        y_neg_fit = fit_slope_neg * x_neg_range
        (self.fit_line_neg,) = ax.plot(
            x_neg_range,
            y_neg_fit,
            "r-",
            linewidth=2,
            animated=True,
            label="Negative x fit",
        )

        # Create text to display fit equations
        self.fit_text = ax.text(
            0.05,
            0.95,
            f"Positive: y = {fit_slope_pos:.3f}x\nNegative: y = {fit_slope_neg:.3f}x",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            verticalalignment="top",
        )
        # Set x and y axis labels
        self.ax.set_xlabel("V_predicted * (env_temp - temp_room)")
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
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Calculate the relationship data for plotting and fit slopes for positive and negative x"""

        v_reals = (env_temps - self.prev_env_temps) / self.dt
        v_predicts = -self.move_factor * (body_temps - self.prefer_temp) * gradients**2

        x_data = v_predicts * (env_temps - self.temp_room)
        y_data = v_reals - v_predicts

        # Separate positive and negative x data
        valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))

        # Fit for positive x values
        pos_mask = valid_mask & (x_data > 0)
        fit_slope_pos = 0
        if np.sum(pos_mask) > 5:  # Need at least 5 points for robust fitting
            x_pos = x_data[pos_mask].reshape(-1, 1)
            y_pos = y_data[pos_mask]

            # normalize y and x to make them have normal scale
            scale = np.nanstd(x_pos) + np.nanstd(y_pos)
            if scale > 0:
                x_pos = x_pos / scale
                y_pos = y_pos / scale

            # Use HuberRegressor with no intercept (force through origin)
            huber_pos = HuberRegressor(fit_intercept=False, epsilon=1.35, max_iter=1000)
            huber_pos.fit(x_pos, y_pos)
            fit_slope_pos = huber_pos.coef_[0]

        # Fit for negative x values
        neg_mask = valid_mask & (x_data < 0)
        fit_slope_neg = 0
        if np.sum(neg_mask) > 5:  # Need at least 5 points for robust fitting
            x_neg = x_data[neg_mask].reshape(-1, 1)
            y_neg = y_data[neg_mask]

            # normalize y and x to make them have normal scale
            scale = np.nanstd(x_neg) + np.nanstd(y_neg)
            if scale > 0:
                x_neg = x_neg / scale
                y_neg = y_neg / scale

            # Use HuberRegressor with no intercept (force through origin)
            huber_neg = HuberRegressor(fit_intercept=False, epsilon=1.35, max_iter=1000)
            huber_neg.fit(x_neg, y_neg)
            fit_slope_neg = huber_neg.coef_[0]

        return x_data, y_data, fit_slope_pos, fit_slope_neg

    def update(
        self, body_temps: np.ndarray, env_temps: np.ndarray, gradients: np.ndarray
    ) -> List[Any]:
        # Calculate new velocity relationship data
        x_data, y_data, fit_slope_pos, fit_slope_neg = (
            self._calculate_velocity_relationship(body_temps, env_temps, gradients)
        )

        # update the max x in history with a weighted average
        self.prev_env_temps = env_temps

        # Update scatter plot data
        self.scatter.set_offsets(np.column_stack([x_data, y_data]))

        # Update fit lines
        x_limit = np.nanmean(self.std_x_in_history) * 10

        # Update positive x fit line (blue)
        x_pos_range = np.array([0, x_limit])
        y_pos_fit = fit_slope_pos * x_pos_range
        self.fit_line_pos.set_data(x_pos_range, y_pos_fit)

        # Update negative x fit line (red)
        x_neg_range = np.array([-x_limit, 0])
        y_neg_fit = fit_slope_neg * x_neg_range
        self.fit_line_neg.set_data(x_neg_range, y_neg_fit)

        # Update text to display fit equations
        self.fit_text.set_text(
            f"Positive: y = {fit_slope_pos:.3f}x\nNegative: y = {fit_slope_neg:.3f}x"
        )

        self._update_axis_limits(x_data, y_data)

        return [self.scatter, self.fit_line_pos, self.fit_line_neg, self.fit_text]
