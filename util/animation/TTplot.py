from typing import Any, Callable, List, Tuple

import numpy as np
from matplotlib.axes import Axes

from util.stable_temp import get_stable_point


class PhasePlot:
    def __init__(
        self,
        ax: Axes,
        body_temps: np.ndarray,
        env_temps: np.ndarray,
        prefer_temp: float,
        heat_gen_coeff: float,
        heat_e2p_coeff: float,
        temp_room: float,
    ) -> None:
        self.ax = ax

        self.x_stable, self.y_stable = get_stable_point(
            prefer_temp, heat_gen_coeff, heat_e2p_coeff
        )

        # Set fixed ranges centered around stable point
        x_range_half = (self.x_stable - temp_room) * 0.05
        y_range_half = (self.y_stable - temp_room) * 0.5

        self.x_range = [self.x_stable - x_range_half, self.x_stable + x_range_half]
        self.y_range = [self.y_stable - y_range_half, self.y_stable + y_range_half / 2]

        # Create scatter plot for body vs environmental temperature
        self.scatter = ax.scatter(
            body_temps, env_temps, edgecolor="gray", s=10, alpha=0.7, animated=True
        )

        # Set up plot properties
        ax.set_xlabel("Body Temperature")
        ax.set_ylabel("Environmental Temperature")
        ax.set_xlim(*self.x_range)
        ax.set_ylim(*self.y_range)
        # ax.set_xlim(body_temps.min(), body_temps.max())
        # ax.set_ylim(env_temps.min(), env_temps.max())
        ax.grid(True)
        ax.set_title("Body vs Env Temp")

    def update(self, body_temps: np.ndarray, env_temps: np.ndarray) -> List[Any]:
        # Update scatter plot
        self.scatter.set_offsets(np.column_stack([body_temps, env_temps]))
        # self.ax.set_xlim(body_temps.min(), body_temps.max())
        # self.ax.set_ylim(env_temps.min(), env_temps.max())
        return [self.scatter]


class VectorFieldPlot:
    def __init__(
        self,
        ax: Axes,
        grad_temps: np.ndarray,
        gradients: np.ndarray,
        vector_field_func: Callable,
        prefer_temp: float,
        heat_gen_coeff: float,
        heat_e2p_coeff: float,
        temp_room: float,
    ) -> None:
        self.ax = ax
        self.vector_field_func = vector_field_func
        self.prefer_temp = prefer_temp

        # Calculate theoretical stable point
        self.x_stable, self.y_stable = get_stable_point(
            prefer_temp, heat_gen_coeff, heat_e2p_coeff
        )

        # Set fixed ranges centered around stable point
        x_range_half = (self.x_stable - temp_room) * 0.03
        y_range_half = (self.y_stable - temp_room) * 0.6

        self.x_range = [self.x_stable - x_range_half, self.x_stable + x_range_half]
        self.y_range = [self.y_stable - y_range_half, self.y_stable + y_range_half / 2]

        # Create fixed vector field grid
        x_vec = np.linspace(self.x_range[0], self.x_range[1], 100)
        y_vec = np.linspace(self.y_range[0], self.y_range[1], 100)
        self.Xs, self.Ys = np.meshgrid(x_vec, y_vec)

        # Calculate initial vector field
        U_vec, V_vec = vector_field_func(self.Xs, self.Ys, grad_temps, gradients)
        U_vec, V_vec = self.normalize_vector_field(U_vec, V_vec)

        # Create vector field
        self.step = 3
        self.quiver = ax.quiver(
            self.Xs[:: self.step, :: self.step],
            self.Ys[:: self.step, :: self.step],
            U_vec[:: self.step, :: self.step],
            V_vec[:: self.step, :: self.step],
            animated=True,
            scale=10,
            width=0.002,
        )

        # Plot isoclines
        ax.axvline(self.x_stable, color="blue", linestyle="--", alpha=0.8)
        ax.plot(
            x_vec,
            x_vec - heat_gen_coeff / heat_e2p_coeff,
            color="red",
            linestyle="--",
            alpha=0.8,
        )

        # Set fixed plot limits
        ax.set_xlim(*self.x_range)
        ax.set_ylim(*self.y_range)

    def normalize_vector_field(
        self, U_vec: np.ndarray, V_vec: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Normalize the vector field to fit the plot view
        # Compute the plot width and height
        x_width = self.x_range[1] - self.x_range[0]
        y_height = self.y_range[1] - self.y_range[0]

        U_vec = U_vec / x_width
        V_vec = V_vec / y_height

        # print(abs(U_vec.max() - U_vec.min()), abs(V_vec.max() - V_vec.min()))
        # print(x_width, y_height)

        return U_vec, V_vec

    def update(
        self,
        grad_temps: np.ndarray,
        gradients: np.ndarray,
    ) -> List[Any]:
        # Update vector field using fixed grid
        U_vec, V_vec = self.vector_field_func(self.Xs, self.Ys, grad_temps, gradients)
        U_vec, V_vec = self.normalize_vector_field(U_vec, V_vec)

        # Update quiver
        self.quiver.set_UVC(
            U_vec[:: self.step, :: self.step], V_vec[:: self.step, :: self.step]
        )

        return [self.quiver]


class TheoryEvolutionPlot:
    """理論演化系統的可視化類，包含向量場和企鵝點的演化"""

    def __init__(
        self,
        ax: Axes,
        theory_evolution,
        prefer_temp: float,
        heat_gen_coeff: float,
        heat_e2p_coeff: float,
        temp_room: float,
    ) -> None:
        self.ax = ax
        self.theory_evolution = theory_evolution
        self.prefer_temp = prefer_temp

        # Calculate theoretical stable point
        self.x_stable, self.y_stable = get_stable_point(
            prefer_temp, heat_gen_coeff, heat_e2p_coeff
        )

        # Set plot ranges
        x_range_half = max(20, abs(self.x_stable - temp_room) * 0.8)
        y_range_half = max(30, abs(self.y_stable - temp_room) * 0.8)

        self.x_range = [self.x_stable - x_range_half, self.x_stable + x_range_half]
        self.y_range = [self.y_stable - y_range_half, self.y_stable + y_range_half]

        # Create vector field grid
        x_vec = np.linspace(self.x_range[0], self.x_range[1], 20)
        y_vec = np.linspace(self.y_range[0], self.y_range[1], 20)
        self.Xs, self.Ys = np.meshgrid(x_vec, y_vec)

        # Calculate vector field
        X, Y, DX, DY = theory_evolution.get_vector_field(x_vec, y_vec)

        # Normalize vectors for better visualization
        magnitude = np.sqrt(DX**2 + DY**2)
        magnitude_nonzero = np.maximum(magnitude, 1e-6)
        scale_factor = 0.8
        DX_norm = DX / magnitude_nonzero * scale_factor
        DY_norm = DY / magnitude_nonzero * scale_factor

        # Create quiver plot for vector field
        self.quiver = ax.quiver(
            X,
            Y,
            DX_norm,
            DY_norm,
            magnitude,
            cmap="viridis",
            alpha=0.6,
            animated=True,
            scale=15,
            width=0.003,
        )

        # Get initial penguin states
        x_penguins, y_penguins = theory_evolution.get_state()

        # Create scatter plot for penguins
        self.scatter = ax.scatter(
            x_penguins, y_penguins, c="red", s=8, alpha=0.7, animated=True
        )

        # Add stable point and isoclines
        ax.plot(self.x_stable, self.y_stable, "ko", markersize=8, label="Stable Point")
        ax.axvline(
            self.prefer_temp,
            color="blue",
            linestyle="--",
            alpha=0.5,
            label="Prefer Temp",
        )
        ax.axhline(0, color="gray", linestyle="-", alpha=0.3)

        # Set plot properties
        ax.set_xlabel("Body Temperature (x)")
        ax.set_ylabel("Environmental Temperature (y)")
        ax.set_xlim(*self.x_range)
        ax.set_ylim(*self.y_range)
        ax.grid(True, alpha=0.3)
        ax.set_title("Theoretical Evolution with Vector Field")
        ax.legend(loc="upper right", fontsize=8)

    def update(self) -> List[Any]:
        """更新理論演化圖"""
        # Get current penguin states
        x_penguins, y_penguins = self.theory_evolution.get_state()

        # Update scatter plot
        self.scatter.set_offsets(np.column_stack([x_penguins, y_penguins]))

        # The vector field remains static, so we don't need to update the quiver

        return [self.scatter, self.quiver]
