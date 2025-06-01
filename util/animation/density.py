from typing import Any, List

import numpy as np
from matplotlib.axes import Axes

from util import update_y_limits


class DensityPlot:
    def __init__(
        self,
        ax: Axes,
        temps: np.ndarray,
        densities: np.ndarray,
        temp_room: float,
        prefer_temp_common: float,
    ) -> None:
        self.ax = ax

        # Create the density line plot
        (self.line,) = ax.plot(temps, densities, animated=True, label="Actual")

        # Set up plot properties
        ax.set_xlim(temp_room, prefer_temp_common)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Penguin Density (penguins/unit²)")
        ax.set_title("Temperature vs Penguin Density")
        ax.legend()

    def update(self, temps: np.ndarray, densities: np.ndarray) -> List[Any]:
        # Update line data
        self.line.set_data(temps, densities)

        # Update axis limits based on data
        update_y_limits(self.ax, densities)

        self.ax.set_xlim(temps.min(), temps.max())

        return [self.line]
