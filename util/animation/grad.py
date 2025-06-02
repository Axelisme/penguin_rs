from typing import Any, List

import numpy as np
from matplotlib.axes import Axes
from scipy.optimize import curve_fit


def quadratic_func(x, a, x0, c):
    """Quadratic function for curve fitting: y = a(x-x0)² + c with non-negative constraint"""
    result = a * (x - x0) ** 2 + c
    return np.maximum(result, 0)  # Ensure non-negative values


class GradPlot:
    def __init__(
        self,
        ax: Axes,
        temps: np.ndarray,
        gradients: np.ndarray,
        temp_room: float,
        prefer_temp: float,
    ) -> None:
        self.ax = ax

        self.max_y_in_history = np.max(gradients)

        # Create the gradient line plot
        (self.line,) = ax.plot(
            temps, gradients, animated=True, label="Actual", color="blue"
        )

        # Create the quadratic fit line plot
        (self.fit_line,) = ax.plot(
            [], [], animated=True, label="Quadratic Fit", color="red", linestyle="--"
        )

        # Create text annotation for displaying parameters
        self.param_text = ax.text(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            fontsize=10,
            animated=True,
        )

        # Set up plot properties
        ax.set_xlim(temp_room, prefer_temp)
        ax.set_ylim(0, self.max_y_in_history * 1.1 + 1e-6)
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Average Gradient")
        ax.set_title("Temperature vs Gradient Relationship")
        ax.legend()

    def update(self, temps: np.ndarray, gradients: np.ndarray) -> List[Any]:
        # Update line data
        self.line.set_data(temps, gradients)
        self.max_y_in_history = max(np.max(gradients), self.max_y_in_history)

        # Perform quadratic fitting
        param_text = "Fit: Failed"
        try:
            # Initial guess for parameters: [a, x0, c]
            # a: negative for downward opening parabola
            # x0: temperature at maximum gradient (optimal temperature)
            # c: maximum gradient value
            temp_at_max = temps[np.argmax(gradients)]
            p0 = [
                np.max(gradients) / (temps.max() - temps.min()) ** 2,
                temp_at_max,
                np.max(gradients),
            ]

            # Perform curve fitting
            popt, _ = curve_fit(quadratic_func, temps, gradients, p0=p0, maxfev=1000)
            a, x0, c = popt

            # Generate smooth curve for plotting
            temp_smooth = np.linspace(temps.min(), temps.max(), 100)
            gradient_fit = quadratic_func(temp_smooth, *popt)

            # Update fit line
            self.fit_line.set_data(temp_smooth, gradient_fit)

            # Format parameter text
            param_text = f"Fit: {a:.4f}(x {-x0:+.2f})² + {c:.4f}"

        except (RuntimeError, ValueError):
            # If fitting fails, clear the fit line
            self.fit_line.set_data([], [])

        # Update parameter text
        self.param_text.set_text(param_text)

        # Update axis limits based on data
        self.ax.set_ylim(0, self.max_y_in_history * 1.1 + 1e-6)

        # Set x-axis limits based on data range with some padding
        temp_range = temps.max() - temps.min()
        padding = temp_range * 0.1
        self.ax.set_xlim(temps.min() - padding, temps.max() + padding)

        return [self.line, self.fit_line, self.param_text]
