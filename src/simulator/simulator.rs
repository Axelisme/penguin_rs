use ndarray::Array2;

pub struct Penguin {
    pub pos: [f64; 2],
    pub vel: [f64; 2],
    pub body_temp: f64,
    pub prefer_temp: f64,
}

impl Penguin {
    pub fn new(pos: [f64; 2], vel: [f64; 2], body_temp: f64, prefer_temp: f64) -> Self {
        Self {
            pos,
            vel,
            body_temp,
            prefer_temp,
        }
    }
}

pub struct AirTemp {
    pub temp: Array2<f64>,
    pub size_x: f64,
    pub size_y: f64,
    pub deffusion_coeff: f64,
    pub decay_coeff: f64,
    pub temp_room: f64,
}

impl AirTemp {
    pub fn new(
        init_temp: Array2<f64>,
        box_size: f64,
        deffusion_coeff: f64,
        decay_coeff: f64,
        temp_room: f64,
    ) -> Self {
        Self {
            temp: init_temp,
            size_x: box_size,
            size_y: box_size,
            deffusion_coeff,
            decay_coeff,
            temp_room,
        }
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

        // Clamp indices to avoid out-of-bounds
        let i_left = (i + nx - 1) % nx;
        let i_right = (i + 1) % nx;
        let j_down = (j + ny - 1) % ny;
        let j_up = (j + 1) % ny;

        let t_left = self.temp[(i_left, j)];
        let t_right = self.temp[(i_right, j)];
        let t_down = self.temp[(i, j_down)];
        let t_up = self.temp[(i, j_up)];

        let grad_x = (t_right - t_left) / (2. * grid_x);
        let grad_y = (t_up - t_down) / (2. * grid_y);

        [grad_x, grad_y]
    }

    pub fn get_derive(&self, heat_src: Array2<f64>) -> Array2<f64> {
        let (nx, ny) = self.temp.dim();
        let grid_x = self.size_x / nx as f64;
        let grid_y = self.size_y / ny as f64;

        let laplacian = {
            let mut laplacian = Array2::<f64>::zeros(self.temp.dim());
            for i in 0..nx {
                for j in 0..ny {
                    let t_center = self.temp[(i, j)];
                    let t_left = self.temp[(i, (j + ny - 1) % ny)];
                    let t_right = self.temp[(i, (j + 1) % ny)];
                    let t_down = self.temp[((i + nx - 1) % nx, j)];
                    let t_up = self.temp[((i + 1) % nx, j)];
                    laplacian[(i, j)] =
                        (t_left + t_right + t_up + t_down - 4.0 * t_center) / (grid_x * grid_y);
                }
            }
            laplacian
        };

        return heat_src + self.deffusion_coeff * laplacian
            - self.decay_coeff * (self.temp.clone() - self.temp_room);
    }
}

pub struct SimulationConfig {
    penguin_move_factor: f64,
    penguin_radius: f64,
    heat_gen_coeff: f64,
    heat_p2e_coeff: f64,
    heat_e2p_coeff: f64,
    enable_collision: bool,
}

impl SimulationConfig {
    pub fn new(
        penguin_move_factor: f64,
        penguin_radius: f64,
        heat_gen_coeff: f64,
        heat_p2e_coeff: f64,
        heat_e2p_coeff: f64,
        enable_collision: bool,
    ) -> Self {
        Self {
            penguin_move_factor,
            penguin_radius,
            heat_gen_coeff,
            heat_p2e_coeff,
            heat_e2p_coeff,
            enable_collision,
        }
    }
}

pub struct Simulation {
    pub config: SimulationConfig,
    pub penguins: Vec<Penguin>,
    pub air: AirTemp,
}

impl Simulation {
    pub fn new(
        config: SimulationConfig,
        init_penguins: Vec<Penguin>,
        init_air_temp: AirTemp,
    ) -> Self {
        Self {
            config,
            penguins: init_penguins,
            air: init_air_temp,
        }
    }

    fn get_derive(&self) -> (Vec<f64>, Array2<f64>, Vec<[f64; 2]>) {
        let t_airs = self
            .penguins
            .iter()
            .map(|p| self.air.get_temp(p.pos))
            .collect::<Vec<_>>();

        // heat env->penguin
        let gen_coeff = self.config.heat_gen_coeff;
        let e2p_coeff = self.config.heat_e2p_coeff;
        let derive_body_temps = self
            .penguins
            .iter()
            .zip(t_airs.iter())
            .map(|(p, t_air)| gen_coeff - e2p_coeff * (p.body_temp - t_air))
            .collect::<Vec<_>>();

        // heat penguin->env
        let derive_air_temp = {
            let heat_r = self.config.penguin_radius;
            let p2e_coeff = self.config.heat_p2e_coeff;

            let mut heat_src = Array2::<f64>::zeros(self.air.temp.dim());
            let (nx, ny) = self.air.temp.dim();
            let grid_x = self.air.size_x / nx as f64;
            let grid_y = self.air.size_y / ny as f64;
            let x_num = (5. * heat_r / grid_x).ceil() as i32;
            let y_num = (5. * heat_r / grid_y).ceil() as i32;
            self.penguins
                .iter()
                .zip(t_airs.iter())
                .for_each(|(p, t_air)| {
                    let [x, y] = p.pos;

                    let (i, j) = self.air.pos2idx(p.pos);
                    for di in -x_num..=x_num {
                        for dj in -y_num..=y_num {
                            let gi = (i as i32 + di + nx as i32) as usize % nx;
                            let gj = (j as i32 + dj + ny as i32) as usize % ny;
                            let [gx, gy] = self.air.idx2pos(gi, gj);
                            let dist2 = (gx - x) * (gx - x) + (gy - y) * (gy - y);
                            heat_src[(gi, gj)] += p2e_coeff
                                * (p.body_temp - t_air)
                                * (-0.5 * dist2 / (heat_r * heat_r)).exp();
                        }
                    }
                });
            self.air.get_derive(heat_src)
        };

        // compute velocities based on temperature gradient
        let vel_factor = self.config.penguin_move_factor;
        let velocities = self
            .penguins
            .iter()
            .enumerate()
            .map(|(_, p)| {
                let [gx, gy] = self.air.get_grad(p.pos);
                let g_norm = (gx * gx + gy * gy).sqrt();
                if g_norm > 1e-6 {
                    let speed = vel_factor * (p.body_temp - p.prefer_temp);
                    [-speed * gx, -speed * gy]
                } else {
                    return [0., 0.];
                }
            })
            .collect::<Vec<_>>();

        (derive_body_temps, derive_air_temp, velocities)
    }

    fn apply_colli(&self, positions: Vec<[f64; 2]>) -> Vec<[f64; 2]> {
        // Optimized brute force collision detection
        let colli_dist = 2.0 * self.config.penguin_radius;
        let colli_dist_sq = colli_dist * colli_dist;
        let n = positions.len();

        // Pre-allocate correction vectors for better performance
        let mut corrections = vec![[0.0; 2]; n];

        // Brute force collision detection with optimizations
        for i in 0..n {
            for j in (i + 1)..n {
                // Calculate distance vector
                let dx = positions[i][0] - positions[j][0];
                let dy = positions[i][1] - positions[j][1];
                let dist_sq = dx * dx + dy * dy;

                // Early exit if no collision or same position
                if dist_sq >= colli_dist_sq || dist_sq < 1e-12 {
                    continue;
                }

                // Calculate collision response
                let dist = dist_sq.sqrt();
                let overlap = colli_dist - dist;
                let correction_magnitude = 0.5 * overlap / dist;

                // Apply correction forces
                let correction_x = correction_magnitude * dx;
                let correction_y = correction_magnitude * dy;

                // Penguin i gets pushed away from j
                corrections[i][0] += correction_x;
                corrections[i][1] += correction_y;

                // Penguin j gets pushed away from i
                corrections[j][0] -= correction_x;
                corrections[j][1] -= correction_y;
            }
        }

        // Apply corrections and boundary conditions
        positions
            .iter()
            .zip(corrections.iter())
            .map(|(pos, correction)| {
                let new_x = pos[0] + correction[0];
                let new_y = pos[1] + correction[1];
                [
                    new_x.rem_euclid(self.air.size_x),
                    new_y.rem_euclid(self.air.size_y),
                ]
            })
            .collect()
    }

    fn apply_colli_and_update(
        &self,
        positions: Vec<[f64; 2]>,
        velocities: Vec<[f64; 2]>,
        dt: f64,
    ) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
        // Update positions based on new velocities
        let mut positions = positions
            .iter()
            .zip(velocities.iter())
            .map(|(p, v)| [p[0] + v[0] * dt, p[1] + v[1] * dt])
            .collect::<Vec<_>>();

        // Apply boundary conditions
        positions.iter_mut().for_each(|pos| {
            pos[0] = pos[0].rem_euclid(self.air.size_x);
            pos[1] = pos[1].rem_euclid(self.air.size_y);
        });

        positions = if self.config.enable_collision {
            self.apply_colli(positions)
        } else {
            positions
        };
        (positions, velocities)
    }

    pub fn step(&mut self, dt: f64) {
        // get derivatives
        let (derive_body_temps, derive_air_temp, velocities) = self.get_derive();

        // apply collision and update positions
        let positions = self.penguins.iter().map(|p| p.pos).collect::<Vec<_>>();
        let (new_positions, velocities) = self.apply_colli_and_update(positions, velocities, dt);

        // update penguin info
        self.penguins.iter_mut().enumerate().for_each(|(i, p)| {
            p.pos = new_positions[i];
            p.vel = velocities[i];
            p.body_temp = derive_body_temps[i] * dt + p.body_temp;
        });

        // update air temp
        self.air
            .temp
            .iter_mut()
            .zip(derive_air_temp.iter())
            .for_each(|(t, derive_air_temp)| {
                *t += *derive_air_temp * dt;
            });
    }
}
