import numpy as np
from numba import jit, prange


@jit(nopython=True, parallel=True)
def _compute_second_order_upwind_divergence_x(jx, vx, dx):
    """
    Compute second-order upwind divergence in x-direction
    Optimized with numba for performance
    """
    ny, nx = jx.shape
    div_jx = np.zeros_like(jx)

    for j in prange(ny):
        for i in range(nx):
            if vx[j, i] > 0:  # Forward difference (upwind)
                if i >= 2:
                    # Second-order upwind: (3*f[i] - 4*f[i-1] + f[i-2]) / (2*dx)
                    div_jx[j, i] = (3 * jx[j, i] - 4 * jx[j, i - 1] + jx[j, i - 2]) / (
                        2 * dx
                    )
                elif i == 1:
                    # First-order upwind
                    div_jx[j, i] = (jx[j, i] - jx[j, i - 1]) / dx
                else:  # i == 0
                    # Forward difference
                    div_jx[j, i] = (jx[j, i + 1] - jx[j, i]) / dx
            else:  # Backward difference (upwind)
                if i <= nx - 3:
                    # Second-order upwind: (-3*f[i] + 4*f[i+1] - f[i+2]) / (2*dx)
                    div_jx[j, i] = (-3 * jx[j, i] + 4 * jx[j, i + 1] - jx[j, i + 2]) / (
                        2 * dx
                    )
                elif i == nx - 2:
                    # First-order upwind
                    div_jx[j, i] = (jx[j, i + 1] - jx[j, i]) / dx
                else:  # i == nx - 1
                    # Backward difference
                    div_jx[j, i] = (jx[j, i] - jx[j, i - 1]) / dx

    return div_jx


@jit(nopython=True, parallel=True)
def _compute_second_order_upwind_divergence_y(jy, vy, dy):
    """
    Compute second-order upwind divergence in y-direction
    Optimized with numba for performance
    """
    ny, nx = jy.shape
    div_jy = np.zeros_like(jy)

    for j in prange(ny):
        for i in range(nx):
            if vy[j, i] > 0:  # Forward difference (upwind)
                if j >= 2:
                    # Second-order upwind: (3*f[j] - 4*f[j-1] + f[j-2]) / (2*dy)
                    div_jy[j, i] = (3 * jy[j, i] - 4 * jy[j - 1, i] + jy[j - 2, i]) / (
                        2 * dy
                    )
                elif j == 1:
                    # First-order upwind
                    div_jy[j, i] = (jy[j, i] - jy[j - 1, i]) / dy
                else:  # j == 0
                    # Forward difference
                    div_jy[j, i] = (jy[j + 1, i] - jy[j, i]) / dy
            else:  # Backward difference (upwind)
                if j <= ny - 3:
                    # Second-order upwind: (-3*f[j] + 4*f[j+1] - f[j+2]) / (2*dy)
                    div_jy[j, i] = (-3 * jy[j, i] + 4 * jy[j + 1, i] - jy[j + 2, i]) / (
                        2 * dy
                    )
                elif j == ny - 2:
                    # First-order upwind
                    div_jy[j, i] = (jy[j + 1, i] - jy[j, i]) / dy
                else:  # j == ny - 1
                    # Backward difference
                    div_jy[j, i] = (jy[j, i] - jy[j - 1, i]) / dy

    return div_jy


class TheoreticalEvolution:
    """理論演化系統模擬器 (基於密度)"""

    def __init__(
        self,
        init_density,
        xs,
        ys,
        grad_func,
        heat_gen,
        heat_e2p,
        move_factor,
        prefer_temp,
    ):
        """
        init_density: shape: (num_y, num_x)
        xs: shape: (num_x,)
        ys: shape: (num_y,)
        """

        self.xs = xs
        self.ys = ys
        self.density = self.normalize_density(init_density)

        self.X, self.Y = np.meshgrid(xs, ys)

        self.grad_func = grad_func
        self.heat_gen = heat_gen
        self.heat_e2p = heat_e2p
        self.move_factor = move_factor
        self.prefer_temp = prefer_temp

        # pre-calculate grad2
        self.cache_grad2 = self.grad_func(self.Y)

    def dxdt(self, density, x, y):
        return self.heat_gen - self.heat_e2p * (x - y)

    def dydt(self, density, x, y):
        return -self.move_factor * (x - self.prefer_temp) * self.cache_grad2

    def calculate_stable_dt(self, vx, vy):
        # make sure vx*dt < dx and vy*dt < dy
        dx = self.xs[1] - self.xs[0]
        dy = self.ys[1] - self.ys[0]
        dt = min(dx / np.max(np.abs(vx)), dy / np.max(np.abs(vy)))
        return dt / 10

    def normalize_density(self, density):
        # set boundary to 0
        density[0, :] = 0
        density[-1, :] = 0
        density[:, 0] = 0
        density[:, -1] = 0

        density = np.clip(density, 0.0, None)
        # Avoid division by zero if sum is zero
        current_sum = (
            np.sum(density) * (self.xs[1] - self.xs[0]) * (self.ys[1] - self.ys[0])
        )
        if current_sum > 1e-9:  # A small threshold to avoid floating point issues
            density /= current_sum
        return density

    def second_order_upwind_divergence(self, jx, jy, vx, vy, dx, dy):
        """
        Calculate divergence using second-order upwind scheme

        For advection equation: ∂ρ/∂t + ∇·(ρv) = 0
        We need to compute ∇·j = ∂jx/∂x + ∂jy/∂y where j = ρv

        Second-order upwind uses:
        - If v > 0: ∂j/∂x ≈ (3*j[i] - 4*j[i-1] + j[i-2]) / (2*dx)
        - If v < 0: ∂j/∂x ≈ (-3*j[i] + 4*j[i+1] - j[i+2]) / (2*dx)

        Optimized with numba for better performance
        """
        # Use numba-optimized functions for the heavy computation
        div_jx = _compute_second_order_upwind_divergence_x(jx, vx, dx)
        div_jy = _compute_second_order_upwind_divergence_y(jy, vy, dy)

        return div_jx + div_jy

    def upwind_update(self, density, jx, jy, vx, vy, dt) -> np.ndarray:
        """
        Update density using second-order upwind scheme
        """
        dx = self.xs[1] - self.xs[0]
        dy = self.ys[1] - self.ys[0]

        # Use second-order upwind divergence
        div_j = self.second_order_upwind_divergence(jx, jy, vx, vy, dx, dy)

        return density - dt * div_j

    def sub_step(self, vx, vy, dt) -> None:
        jx = self.density * vx
        jy = self.density * vy

        correction_term = np.mean(jy, axis=1)
        correction_term -= correction_term[0]

        jy = jy - correction_term[:, None]

        new_density = self.upwind_update(self.density, jx, jy, vx, vy, dt)

        self.density = self.normalize_density(new_density)

    def step(self, dt) -> None:
        # calculate v using self.density at the START of this dt step
        vx = self.dxdt(self.density, self.X, self.Y)
        vy = self.dydt(self.density, self.X, self.Y)

        stable_dt = self.calculate_stable_dt(vx, vy)

        t = 0
        while t < dt:
            sub_dt = min(stable_dt, dt - t)
            self.sub_step(vx, vy, sub_dt)
            t += sub_dt
