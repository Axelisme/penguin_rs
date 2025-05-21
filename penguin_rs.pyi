# penguin_rs/__init__.pyi
from typing import List, Tuple

import numpy as np

class PySimulation:
    """
    企鵝群體模擬主類別。
    """
    def __init__(
        self,
        init_penguins: np.ndarray[np.float64, np.ndim[2]],
        init_air_temp: np.ndarray[np.float64, np.ndim[2]],
        penguin_max_vel: float,
        penguin_radius: float,
        heat_gen_coeff: float,
        heat_p2e_coeff: float,
        heat_e2p_coeff: float,
        prefer_temp_common: float,
        box_size: float,
        deffusion_coeff: float,
        decay_coeff: float,
        temp_room: float,
    ) -> None: ...
    """
    初始化模擬器。
    參數請參考 Rust 原始碼或 run.py 範例。
    """

    def get_state(
        self,
    ) -> Tuple[
        List[List[float]], List[List[float]], List[float], List[List[float]]
    ]: ...
    """
    取得目前模擬狀態。
    回傳: (positions, velocities, body_temps, air_temps_vec)
      - positions: List of [x, y]
      - velocities: List of [vx, vy]
      - body_temps: List of float
      - air_temps_vec: List of List of float (2D grid)
    """

    def step(self, dt: float) -> None: ...
    """
    推進模擬 dt 秒。
    """
