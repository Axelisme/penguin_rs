import numpy as np


class TheoreticalEvolution:
    """理論演化系統模擬器"""

    def __init__(
        self, init_x, init_y, grad_func, heat_gen, heat_e2p, move_factor, prefer_temp
    ):
        # 初始化企鵝的理論狀態 (x=body_temp, y=env_temp)
        self.x = np.array(init_x, dtype=float)
        self.y = np.array(init_y, dtype=float)

        self.grad_func = grad_func
        self.heat_gen = heat_gen
        self.heat_e2p = heat_e2p
        self.move_factor = move_factor
        self.prefer_temp = prefer_temp

    def dxdt(self, x, y):
        """體溫演化方程: dx/dt = HEAT_GEN_COEFF - HEAT_E2P_COEFF*(x - y)"""
        return self.heat_gen - self.heat_e2p * (x - y)

    def dydt(self, x, y):
        return -self.move_factor * (x - self.prefer_temp) * self.grad_func(y) ** 2

    def calculate_stable_dt(self, vx, vy):
        # make sure vx*dt < dx and vy*dt < dy
        return 0.01

    def sub_step(self, vx, vy, dt):
        sigma = 1

        yi = self.y[:, None]
        yj = self.y[None, :]

        # pi = \sum_j N(y_i, y_j, sigma)
        pi = np.sum(np.exp(-0.5 * ((yi - yj) / sigma) ** 2), axis=1)

        # vi_corr = \sum_j (vj*pj) / pi
        vi_corr = np.mean(vy * pi) / pi

        vy = vy - vi_corr

        self.y = self.y + vy * dt
        self.x = self.x + vx * dt

    def step(self, dt):
        """使用歐拉方法演化一步"""
        # calculate v using self.density at the START of this dt step
        vx = self.dxdt(self.x, self.y)
        vy = self.dydt(self.x, self.y)

        stable_dt = self.calculate_stable_dt(vx, vy)

        t = 0
        while t < dt:
            sub_dt = min(stable_dt, dt - t)
            self.sub_step(vx, vy, sub_dt)
            t += sub_dt

    def get_vector_field(self, x_grid, y_grid):
        """計算向量場用於繪製"""
        X, Y = np.meshgrid(x_grid, y_grid)
        DX = self.dxdt(X, Y)
        DY = self.dydt(X, Y)
        return X, Y, DX, DY
