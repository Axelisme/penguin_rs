from typing import Any, List

import numpy as np
from matplotlib.axes import Axes


class CheckEnvPlot:
    def __init__(
        self,
        ax: Axes,
        velocities: np.ndarray,
        env_temps: np.ndarray,
        gradients: np.ndarray,
        dt: float,
    ) -> None:
        self.ax = ax
        self.dt = dt
        self.prev_env_temps = env_temps.copy()

        # Store multiple previous temperatures for better derivative estimation
        self.temp_history = [env_temps.copy() for _ in range(3)]
        self.history_index = 0

        # Calculate initial relationship data
        dT_dt_actual, v_dot_grad = self._calculate_relationship(
            velocities, env_temps, gradients
        )

        # Create scatter plot
        self.scatter = ax.scatter(
            v_dot_grad, dT_dt_actual, alpha=0.6, s=8, c="blue", animated=True
        )

        # Create diagonal reference line (y = x)
        # Handle case where arrays might be empty or all NaN
        if len(v_dot_grad) > 0 and not np.all(np.isnan(v_dot_grad)):
            x_min, x_max = np.nanmin(v_dot_grad), np.nanmax(v_dot_grad)
            y_min, y_max = np.nanmin(dT_dt_actual), np.nanmax(dT_dt_actual)
            line_min = min(x_min, y_min, -1)  # Default minimum
            line_max = max(x_max, y_max, 1)  # Default maximum
        else:
            # Default range when no valid data
            line_min, line_max = -5, 5

        (self.ref_line,) = ax.plot(
            [line_min, line_max],
            [line_min, line_max],
            "r--",
            linewidth=2,
            alpha=0.7,
            animated=True,
            label="y = x",
        )

        # Set labels and title
        ax.set_xlabel("V·∇T (Velocity dot Gradient)")
        ax.set_ylabel("dT_env/dt (Environmental Temperature Change)")
        ax.set_title("Check: V·∇T ≈ dT_env/dt (penguin advection)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Calculate correlation coefficient and standard deviation
        if (
            len(v_dot_grad) > 1
            and not np.all(np.isnan(v_dot_grad))
            and not np.all(np.isnan(dT_dt_actual))
        ):
            # Check if there's variance in the data
            if np.std(v_dot_grad) > 1e-10 and np.std(dT_dt_actual) > 1e-10:
                correlation = np.corrcoef(v_dot_grad, dT_dt_actual)[0, 1]
                residuals = dT_dt_actual - v_dot_grad
                std_dev = np.std(residuals)
            else:
                correlation = 0.0
                std_dev = 0.0
        else:
            correlation = 0.0
            std_dev = 0.0

        self.stats_text = ax.text(
            0.05,
            0.95,
            f"Correlation: {correlation:.3f}\nStd Dev: {std_dev:.3f}",
            transform=ax.transAxes,
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            verticalalignment="top",
        )

    def _calculate_relationship(
        self,
        velocities: np.ndarray,
        env_temps: np.ndarray,
        gradients: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the relationship between V·∇T and dT_env/dt with improved accuracy"""

        # Use central difference for better derivative estimation if we have enough history
        if len(self.temp_history) >= 3 and self.history_index >= 2:
            # Central difference: f'(t) ≈ [f(t+h) - f(t-h)] / 2h
            current_temps = env_temps
            prev_prev_temps = self.temp_history[(self.history_index - 2) % 3]

            # Use central difference with 2*dt spacing for better accuracy
            dT_dt_actual = (current_temps - prev_prev_temps) / (2.0 * self.dt)
        else:
            # Fallback to forward difference or zero for very first steps
            if len(env_temps) == len(self.prev_env_temps):
                dT_dt_actual = (env_temps - self.prev_env_temps) / self.dt
            else:
                # For very first initialization, return zeros
                dT_dt_actual = np.zeros_like(env_temps)

        # Calculate V·∇T (velocity dot gradient)
        # Note: penguins move as v = -speed * ∇T, so v·∇T = -speed * |∇T|²
        # In advection equation: dT/dt + v·∇T ≈ 0, so v·∇T ≈ -dT/dt
        # But since v = -speed*∇T, we get v·∇T = -speed*|∇T|² which should match dT_env/dt
        v_dot_grad = np.sum(velocities * gradients, axis=1)

        # Filter out extreme outliers (more than 3 standard deviations)
        if len(dT_dt_actual) > 10:
            mean_dT = np.mean(dT_dt_actual)
            std_dT = np.std(dT_dt_actual)
            mean_v_grad = np.mean(v_dot_grad)
            std_v_grad = np.std(v_dot_grad)

            # Only apply outlier filtering if there's sufficient variance
            if std_dT > 1e-10 and std_v_grad > 1e-10:
                # Create mask for reasonable values
                mask = (
                    (np.abs(dT_dt_actual - mean_dT) < 3 * std_dT)
                    & (np.abs(v_dot_grad - mean_v_grad) < 3 * std_v_grad)
                    & (~np.isnan(dT_dt_actual))
                    & (~np.isnan(v_dot_grad))
                )

                dT_dt_actual = dT_dt_actual[mask]
                v_dot_grad = v_dot_grad[mask]

        return dT_dt_actual, v_dot_grad

    def update(
        self,
        velocities: np.ndarray,
        env_temps: np.ndarray,
        gradients: np.ndarray,
    ) -> List[Any]:
        # Update temperature history
        self.history_index = (self.history_index + 1) % 3
        self.temp_history[self.history_index] = env_temps.copy()

        # Calculate new relationship data
        dT_dt_actual, v_dot_grad = self._calculate_relationship(
            velocities, env_temps, gradients
        )

        # Update previous temperatures for next iteration
        self.prev_env_temps = env_temps.copy()

        # Update scatter plot data
        if len(dT_dt_actual) > 0 and len(v_dot_grad) > 0:
            self.scatter.set_offsets(np.column_stack([v_dot_grad, dT_dt_actual]))

            # Update reference line limits
            if not np.all(np.isnan(v_dot_grad)) and not np.all(np.isnan(dT_dt_actual)):
                x_min, x_max = np.nanmin(v_dot_grad), np.nanmax(v_dot_grad)
                y_min, y_max = np.nanmin(dT_dt_actual), np.nanmax(dT_dt_actual)
                line_min = min(x_min, y_min, -1)
                line_max = max(x_max, y_max, 1)
            else:
                line_min, line_max = -5, 5

            self.ref_line.set_data([line_min, line_max], [line_min, line_max])

            # Calculate and update statistics
            if (
                len(v_dot_grad) > 1
                and not np.all(np.isnan(v_dot_grad))
                and not np.all(np.isnan(dT_dt_actual))
            ):
                # Check if there's variance in the data
                if np.std(v_dot_grad) > 1e-10 and np.std(dT_dt_actual) > 1e-10:
                    correlation = np.corrcoef(v_dot_grad, dT_dt_actual)[0, 1]
                    residuals = dT_dt_actual - v_dot_grad
                    std_dev = np.std(residuals)
                else:
                    correlation = 0.0
                    std_dev = 0.0
            else:
                correlation = 0.0
                std_dev = 0.0

            self.stats_text.set_text(
                f"Correlation: {correlation:.3f}\nStd Dev: {std_dev:.3f}"
            )

        # Update axis limits with better scaling
        self.ax.set_xlim(-8, 8)
        self.ax.set_ylim(-8, 8)

        return [self.scatter, self.ref_line, self.stats_text]
