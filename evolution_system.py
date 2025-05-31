import matplotlib.pyplot as plt
import numpy as np

from util import plot_integrated_analysis, simulate_trajectories


def main():
    """
    Main function: demonstrate integrated analysis with single comprehensive plot
    """
    # Set parameters
    A = 0.15
    B = 0.01
    x0 = 20.0  # New parameter x0
    alpha = 0.05

    def grad_func(y: float) -> float:
        # return 1.4 * np.exp(-0.5 * ((y + 5) / 3) ** 2)
        peak_y = -5
        return 1.4 * np.where(
            y < peak_y, (y + 30) / (peak_y + 30), (15 - y) / (15 - peak_y)
        )

    def density_factor(y: float) -> float:
        return 0.5 + np.arctan((y + 5) / 0.5) / np.pi
        # return np.full_like(y, 0.0)

    def H_func(x: float, y: float) -> float:
        return -alpha * (1.0 - density_factor(y)) * grad_func(y) ** 2

    x_range = np.linspace(19, 21, 30)
    y_range = np.linspace(1, 8, 30)
    # x_range = np.linspace(14, 16, 30)
    # y_range = np.linspace(0.1, 0.13, 30)

    # Set initial points for trajectory simulation
    initial_points = [
        (i, j) for i in np.linspace(19, 21, 10) for j in np.linspace(1, 8, 30)
    ]
    # initial_points.append((25, 5))

    print("Two-variable evolution system analysis")
    print(f"dx/dt = {A} - {B}(x-y)")
    print(f"dy/dt = (x-{x0})*H(x,y)")
    print("-" * 50)

    # plot the H function
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(y_range, grad_func(y_range), label="grad_func")
    ax1.plot(y_range, density_factor(y_range), label="density_factor")
    ax1.legend()
    ax1.set_xlabel("y")
    ax1.set_title("H(x,y)")

    possible_ys = np.linspace(0.0, 10, 100)
    ax2.plot(np.cumsum(1 / grad_func(possible_ys)), possible_ys)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 10)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Inverse integral of grad_func")
    plt.show()

    # Simulate trajectories
    t_span = (0, 1000)  # Simulate for 5000 time units
    trajectories = simulate_trajectories(
        initial_points, A, B, x0, H_func, t_span, num_points=10000
    )

    # Create integrated analysis plot
    print("\nPlotting integrated analysis...")
    plot_integrated_analysis(
        A,
        B,
        x0,
        H_func,
        trajectories,
        (x_range[0], x_range[-1]),
        (y_range[0], y_range[-1]),
        grid_density=50,
    )


if __name__ == "__main__":
    main()
