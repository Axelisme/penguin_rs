from typing import Any, List

import numpy as np
from matplotlib.axes import Axes


def create_penguin_colors(body_temps, prefer_temp_common):
    """Create binary colors based on temperature preference"""
    return ["white" if temp > prefer_temp_common else "black" for temp in body_temps]


def update_color_limits(im, data, padding_factor=0.1):
    """Update image color limits with padding"""
    data_min, data_max = np.min(data), np.max(data)
    if data_min == data_max:
        data_min -= 0.5
        data_max += 0.5
    im_pad = (data_max - data_min) * padding_factor
    im.set_clim(vmin=data_min - im_pad, vmax=data_max + im_pad)


class PenguinPlot:
    def __init__(
        self,
        ax: Axes,
        positions: np.ndarray,
        body_temps: np.ndarray,
        air_temp_grids: np.ndarray,
        box_size: float,
        prefer_temp_common: float,
        total_frames: int,
    ) -> None:
        self.ax = ax
        self.box_size = box_size
        self.prefer_temp_common = prefer_temp_common

        # Create binary colors based on temperature preference
        penguin_colors = create_penguin_colors(body_temps, prefer_temp_common)

        # Create air temperature heatmap
        self.im = ax.imshow(
            air_temp_grids.T,
            cmap="coolwarm",
            origin="lower",
            extent=[0, box_size, 0, box_size],
            interpolation="none",
            animated=True,
        )

        # Create penguin scatter plot
        self.scatter = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            c=penguin_colors,
            edgecolor="none",
            animated=True,
            s=10,
        )

        self.total_frames = total_frames

        # Set up plot properties
        ax.set_xlim(0, box_size)
        ax.set_ylim(0, box_size)
        ax.set_xlabel("X position")
        ax.set_ylabel("Y position")
        # self.title = ax.set_title(f"Time: 0.00s Frame: 0/{total_frames}")
        self.title = ax.text(
            0.5,
            0.95,
            f"Time: 0.00s Frame: 0/{total_frames}",
            transform=ax.transAxes,
            ha="center",
        )
        ax.set_aspect("equal", adjustable="box")

    def update(
        self,
        positions: np.ndarray,
        body_temps: np.ndarray,
        air_temp_grids: np.ndarray,
        current_sim_time: float,
        frame: int,
    ) -> List[Any]:
        # Create binary colors based on temperature preference
        penguin_colors = create_penguin_colors(body_temps, self.prefer_temp_common)

        # Update air temperature heatmap
        self.im.set_data(air_temp_grids.T)

        # Update penguin positions and colors
        self.scatter.set_offsets(positions)
        self.scatter.set_color(penguin_colors)

        # Update color limits dynamically
        update_color_limits(self.im, air_temp_grids)

        self.title.set_text(
            f"Time: {current_sim_time:.2f}s Frame: {frame}/{self.total_frames}"
        )

        return [self.im, self.scatter, self.title]
