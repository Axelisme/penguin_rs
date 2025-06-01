use ndarray::Array2;

const MIN_GRADIENT_THRESHOLD: f64 = 1e-6;
const MIN_DISTANCE_THRESHOLD: f64 = 1e-12;
const HEAT_RADIUS_FACTOR: f64 = 5.0;

pub struct AirTemp {
    pub temp: Array2<f64>,
    pub size_x: f64,
    pub size_y: f64,
    pub diffusion_coeff: f64,
    pub decay_coeff: f64,
    pub temp_room: f64,
}

impl AirTemp {
    pub fn new(
        init_temp: Array2<f64>,
        box_size: f64,
        diffusion_coeff: f64,
        decay_coeff: f64,
        temp_room: f64,
    ) -> Self {
        Self {
            temp: init_temp,
            size_x: box_size,
            size_y: box_size,
            diffusion_coeff,
            decay_coeff,
            temp_room,
        }
    }

    /// 計算擴散方程的穩定性條件：dt_max = min(Δx², Δy²) / (2 * D)
    pub fn get_max_stable_dt(&self) -> f64 {
        let (nx, ny) = self.temp.dim();
        let grid_x = self.size_x / nx as f64;
        let grid_y = self.size_y / ny as f64;
        let min_grid_sq = (grid_x * grid_x).min(grid_y * grid_y);
        min_grid_sq / (4.0 * self.diffusion_coeff)
    }

    pub fn idx2pos(&self, i: usize, j: usize) -> [f64; 2] {
        let (nx, ny) = self.temp.dim();
        let grid_x = self.size_x / nx as f64;
        let grid_y = self.size_y / ny as f64;
        let i = i % nx;
        let j = j % ny;
        [(i as f64 + 0.5) * grid_x, (j as f64 + 0.5) * grid_y]
    }

    pub fn pos2idx(&self, pos: [f64; 2]) -> (usize, usize) {
        let (nx, ny) = self.temp.dim();
        let grid_x = self.size_x / nx as f64;
        let grid_y = self.size_y / ny as f64;
        let i_x = (pos[0] / grid_x).floor() as isize;
        let i_y = (pos[1] / grid_y).floor() as isize;
        (
            ((i_x + nx as isize) % nx as isize) as usize,
            ((i_y + ny as isize) % ny as isize) as usize,
        )
    }

    pub fn get_temp(&self, pos: [f64; 2]) -> f64 {
        self.temp[self.pos2idx(pos)]
    }

    pub fn get_grad(&self, pos: [f64; 2]) -> [f64; 2] {
        let (nx, ny) = self.temp.dim();
        let grid_x = self.size_x / nx as f64;
        let grid_y = self.size_y / ny as f64;
        let (i, j) = self.pos2idx(pos);

        let i_left = (i + nx - 1) % nx;
        let i_right = (i + 1) % nx;
        let j_down = (j + ny - 1) % ny;
        let j_up = (j + 1) % ny;

        let grad_x = (self.temp[(i_right, j)] - self.temp[(i_left, j)]) / (2.0 * grid_x);
        let grad_y = (self.temp[(i, j_up)] - self.temp[(i, j_down)]) / (2.0 * grid_y);

        [grad_x, grad_y]
    }

    pub fn get_derivative(&self, heat_src: Array2<f64>) -> Array2<f64> {
        let laplacian = self.compute_laplacian();
        heat_src + self.diffusion_coeff * laplacian
            - self.decay_coeff * (self.temp.clone() - self.temp_room)
    }

    fn compute_laplacian(&self) -> Array2<f64> {
        let (nx, ny) = self.temp.dim();
        let grid_x = self.size_x / nx as f64;
        let grid_y = self.size_y / ny as f64;
        let mut laplacian = Array2::<f64>::zeros(self.temp.dim());

        for i in 0..nx {
            for j in 0..ny {
                let t_center = self.temp[(i, j)];
                let t_left = self.temp[(i, (j + ny - 1) % ny)];
                let t_right = self.temp[(i, (j + 1) % ny)];
                let t_down = self.temp[((i + nx - 1) % nx, j)];
                let t_up = self.temp[((i + 1) % nx, j)];

                // 正确的二維拉普拉斯算子：∇²T = ∂²T/∂x² + ∂²T/∂y²
                let d2_dx2 = (t_left + t_right - 2.0 * t_center) / (grid_x * grid_x);
                let d2_dy2 = (t_down + t_up - 2.0 * t_center) / (grid_y * grid_y);
                laplacian[(i, j)] = d2_dx2 + d2_dy2;
            }
        }
        laplacian
    }
}

#[derive(Clone)]
pub struct SimulationConfig {
    pub penguin_move_factor: f64,
    pub penguin_radius: f64,
    pub heat_gen_coeff: f64,
    pub heat_p2e_coeff: f64,
    pub heat_e2p_coeff: f64,
    pub collision_strength: f64,
}

impl SimulationConfig {
    pub fn new(
        penguin_move_factor: f64,
        penguin_radius: f64,
        heat_gen_coeff: f64,
        heat_p2e_coeff: f64,
        heat_e2p_coeff: f64,
        collision_strength: f64,
    ) -> Self {
        Self {
            penguin_move_factor,
            penguin_radius,
            heat_gen_coeff,
            heat_p2e_coeff,
            heat_e2p_coeff,
            collision_strength,
        }
    }
}

pub struct Simulation {
    pub config: SimulationConfig,
    pub positions: Vec<[f64; 2]>,
    pub velocities: Vec<[f64; 2]>,
    pub body_temps: Vec<f64>,
    pub prefer_temps: Vec<f64>,
    pub air: AirTemp,
}

impl Simulation {
    pub fn new(
        config: SimulationConfig,
        positions: Vec<[f64; 2]>,
        velocities: Vec<[f64; 2]>,
        body_temps: Vec<f64>,
        prefer_temps: Vec<f64>,
        air: AirTemp,
    ) -> Self {
        Self {
            config,
            positions,
            velocities,
            body_temps,
            prefer_temps,
            air,
        }
    }

    pub fn num_penguins(&self) -> usize {
        self.positions.len()
    }

    pub fn step(&mut self, dt: f64) {
        self.step_euler(dt);
    }

    pub fn step_euler(&mut self, dt: f64) {
        // 計算企鵝位置的氣溫
        let air_temps = self
            .positions
            .iter()
            .map(|pos| self.air.get_temp(*pos))
            .collect::<Vec<_>>();

        // 計算企鵝體溫的導數
        let body_temp_derivatives =
            self.compute_body_temp_derivatives(&self.body_temps, &air_temps);

        // 計算基本速度（溫度梯度驅動）
        let mut velocities = self.compute_velocities(
            &self.positions,
            &self.body_temps,
            &self.prefer_temps,
            &self.air,
        );

        // 添加碰撞排斥速度
        if self.config.collision_strength > 0.0 {
            let collision_velocities = self.compute_collision_repulsion(&self.positions);
            velocities
                .iter_mut()
                .zip(collision_velocities.iter())
                .for_each(|(v, cv)| {
                    v[0] += cv[0];
                    v[1] += cv[1];
                });
        }

        // 更新企鵝體溫（一次性）
        self.body_temps
            .iter_mut()
            .zip(body_temp_derivatives.iter())
            .for_each(|(body_temp, derivative)| {
                *body_temp += derivative * dt;
            });

        // 計算氣溫更新需要的子步驟數量, 分子步驟更新氣溫
        let max_stable_dt = self.air.get_max_stable_dt();
        let air_sub_steps = (dt / max_stable_dt).ceil().max(1.0) as usize;
        let air_dt = dt / air_sub_steps as f64;
        for _ in 0..air_sub_steps {
            let air_temps = self
                .positions
                .iter()
                .map(|pos| self.air.get_temp(*pos))
                .collect::<Vec<_>>();

            let air_temp_derivative = self.compute_air_temp_derivative(
                &self.positions,
                &self.body_temps,
                &air_temps,
                &self.air,
            );

            self.air.temp += &(&air_temp_derivative * air_dt);
        }

        // 更新位置（使用修正後的速度）
        self.positions
            .iter_mut()
            .zip(velocities.iter())
            .for_each(|(pos, v)| {
                pos[0] = (pos[0] + v[0] * dt).rem_euclid(self.air.size_x);
                pos[1] = (pos[1] + v[1] * dt).rem_euclid(self.air.size_y);
            });

        self.velocities = velocities;
    }

    fn compute_body_temp_derivatives(&self, body_temps: &[f64], air_temps: &[f64]) -> Vec<f64> {
        body_temps
            .iter()
            .zip(air_temps.iter())
            .map(|(body_temp, air_temp)| {
                self.config.heat_gen_coeff - self.config.heat_e2p_coeff * (body_temp - air_temp)
            })
            .collect()
    }

    fn compute_air_temp_derivative(
        &self,
        positions: &[[f64; 2]],
        body_temps: &[f64],
        air_temps: &[f64],
        temp_air: &AirTemp,
    ) -> Array2<f64> {
        let heat_src = self.compute_heat_source(positions, body_temps, air_temps, temp_air);
        temp_air.get_derivative(heat_src)
    }

    fn compute_heat_source(
        &self,
        positions: &[[f64; 2]],
        body_temps: &[f64],
        air_temps: &[f64],
        temp_air: &AirTemp,
    ) -> Array2<f64> {
        let (nx, ny) = temp_air.temp.dim();
        let grid_x = temp_air.size_x / nx as f64;
        let grid_y = temp_air.size_y / ny as f64;

        let heat_radius = self.config.penguin_radius;
        let x_range = (HEAT_RADIUS_FACTOR * heat_radius / grid_x).ceil() as i32;
        let y_range = (HEAT_RADIUS_FACTOR * heat_radius / grid_y).ceil() as i32;

        let mut heat_src = Array2::<f64>::zeros(temp_air.temp.dim());
        for ((pos, body_temp), air_temp) in positions
            .iter()
            .zip(body_temps.iter())
            .zip(air_temps.iter())
        {
            let (i, j) = temp_air.pos2idx(*pos);

            for di in -x_range..=x_range {
                for dj in -y_range..=y_range {
                    let gi = (i as i32 + di + nx as i32) as usize % nx;
                    let gj = (j as i32 + dj + ny as i32) as usize % ny;
                    let grid_pos = temp_air.idx2pos(gi, gj);
                    let dist_sq = Self::distance_squared(pos, &grid_pos);

                    heat_src[(gi, gj)] += self.config.heat_p2e_coeff
                        * (body_temp - air_temp)
                        * (-0.5 * dist_sq / (heat_radius * heat_radius)).exp();
                }
            }
        }
        heat_src
    }

    fn compute_velocities(
        &self,
        positions: &[[f64; 2]],
        body_temps: &[f64],
        prefer_temps: &[f64],
        temp_air: &AirTemp,
    ) -> Vec<[f64; 2]> {
        positions
            .iter()
            .zip(body_temps.iter())
            .zip(prefer_temps.iter())
            .map(|((pos, body_temp), prefer_temp)| {
                let grad = temp_air.get_grad(*pos);
                let grad_magnitude = Self::vector_magnitude(&grad);

                if grad_magnitude > MIN_GRADIENT_THRESHOLD {
                    let speed = self.config.penguin_move_factor * (body_temp - prefer_temp);
                    [-speed * grad[0], -speed * grad[1]]
                } else {
                    [0.0, 0.0]
                }
            })
            .collect()
    }

    fn compute_collision_repulsion(&self, positions: &[[f64; 2]]) -> Vec<[f64; 2]> {
        let n = positions.len();
        let mut repulsion_velocities = vec![[0.0; 2]; n];

        let interaction_radius = 3.0 * self.config.penguin_radius; // 交互半径
        let sigma = self.config.penguin_radius; // 高斯分布的標準差

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }

                let distance_vec = [
                    positions[i][0] - positions[j][0],
                    positions[i][1] - positions[j][1],
                ];
                let distance_sq = Self::vector_magnitude_squared(&distance_vec);
                let distance = distance_sq.sqrt();

                if distance > interaction_radius || distance < MIN_DISTANCE_THRESHOLD {
                    continue;
                }

                // 高斯分布的排斥力強度：exp(-d²/(2σ²))
                let gaussian_factor = (-distance_sq / (2.0 * sigma * sigma)).exp();
                let repulsion_strength = self.config.collision_strength * gaussian_factor;

                // 計算單位方向向量
                let unit_vec = [distance_vec[0] / distance, distance_vec[1] / distance];

                // 添加排斥速度
                repulsion_velocities[i][0] += repulsion_strength * unit_vec[0];
                repulsion_velocities[i][1] += repulsion_strength * unit_vec[1];
            }
        }

        repulsion_velocities
    }

    // Utility functions
    fn distance_squared(pos1: &[f64; 2], pos2: &[f64; 2]) -> f64 {
        let dx = pos1[0] - pos2[0];
        let dy = pos1[1] - pos2[1];
        dx * dx + dy * dy
    }

    fn vector_magnitude(v: &[f64; 2]) -> f64 {
        (v[0] * v[0] + v[1] * v[1]).sqrt()
    }

    fn vector_magnitude_squared(v: &[f64; 2]) -> f64 {
        v[0] * v[0] + v[1] * v[1]
    }
}
