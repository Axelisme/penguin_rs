from typing import Callable, List, Tuple

import numpy as np
from scipy.integrate import odeint


def evolution_system(
    state: List[float], t: float, A: float, B: float, x0: float, H_func: Callable
) -> List[float]:
    """
    Two-variable evolution system
    dx/dt = A - B(x-y)
    dy/dt = (x-x0)*H(x, y)

    Args:
        state: [x, y] current state
        t: time
        A: parameter A
        B: parameter B
        x0: parameter x0 (reference point for dy/dt equation)
        H_func: function H(x, y)

    Returns:
        [dx/dt, dy/dt] state change rate
    """
    x, y = state
    dxdt = A - B * (x - y)
    dydt = (x - x0) * H_func(x, y)
    return [dxdt, dydt]


def simulate_trajectories(
    initial_points: List[Tuple[float, float]],
    A: float,
    B: float,
    x0: float,
    H_func: Callable,
    t_span: Tuple[float, float],
    num_points: int = 1000,
) -> List[np.ndarray]:
    """
    Simulate trajectory evolution from given initial points

    Args:
        initial_points: list of initial points [(x0, y0), ...]
        A: parameter A
        B: parameter B
        x0: parameter x0 (reference point for dy/dt equation)
        H_func: function H(x, y)
        t_span: time range (t_start, t_end)
        num_points: number of time points

    Returns:
        List of trajectory arrays corresponding to each initial point, each array has shape (num_points, 2)
    """
    t = np.linspace(t_span[0], t_span[1], num_points)
    trajectories = []

    for x_init, y_init in initial_points:
        initial_state = [x_init, y_init]
        trajectory = odeint(evolution_system, initial_state, t, args=(A, B, x0, H_func))
        trajectories.append(trajectory)

    return trajectories
