use kdtree::distance::squared_euclidean;
use kdtree::KdTree;
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
        let grid_x = self.size_x / self.temp.shape()[0] as f64;
        let grid_y = self.size_y / self.temp.shape()[1] as f64;
        [(i as f64 + 0.5) * grid_x, (j as f64 + 0.5) * grid_y]
    }

    pub fn pos2idx(&self, pos: [f64; 2]) -> (usize, usize) {
        let grid_x = self.size_x / self.temp.shape()[0] as f64;
        let grid_y = self.size_y / self.temp.shape()[1] as f64;
        (
            ((pos[0] / grid_x).floor() as usize).clamp(0, self.temp.shape()[0] - 1),
            ((pos[1] / grid_y).floor() as usize).clamp(0, self.temp.shape()[1] - 1),
        )
    }

    pub fn get_temp(&self, pos: [f64; 2]) -> f64 {
        self.temp[self.pos2idx(pos)]
    }

    pub fn get_grad(&self, pos: [f64; 2]) -> [f64; 2] {
        let num_grid_x = self.temp.shape()[0];
        let num_grid_y = self.temp.shape()[1];
        let grid_x = self.size_x / num_grid_x as f64;
        let grid_y = self.size_y / num_grid_y as f64;

        let (i, j) = self.pos2idx(pos);

        // Clamp indices to avoid out-of-bounds
        let i_left = if i > 0 { i - 1 } else { i };
        let i_right = if i + 1 < num_grid_x { i + 1 } else { i };
        let j_down = if j > 0 { j - 1 } else { j };
        let j_up = if j + 1 < num_grid_y { j + 1 } else { j };

        let t_left = self.temp[(i_left, j)];
        let t_right = self.temp[(i_right, j)];
        let t_down = self.temp[(i, j_down)];
        let t_up = self.temp[(i, j_up)];

        let grad_x = (t_right - t_left) / (2. * grid_x);
        let grad_y = (t_up - t_down) / (2. * grid_y);

        [grad_x, grad_y]
    }

    pub fn get_derive(&self, heat_src: Array2<f64>) -> Array2<f64> {
        let num_grid_x = self.temp.shape()[0];
        let num_grid_y = self.temp.shape()[1];
        let grid_x = self.size_x / num_grid_x as f64;
        let grid_y = self.size_y / num_grid_y as f64;

        let laplacian = {
            let mut laplacian = Array2::<f64>::zeros(self.temp.dim());
            for i in 1..num_grid_x - 1 {
                for j in 1..num_grid_y - 1 {
                    let t_center = self.temp[(i, j)];
                    let t_left = self.temp[(i, j - 1)];
                    let t_right = self.temp[(i, j + 1)];
                    let t_down = self.temp[(i - 1, j)];
                    let t_up = self.temp[(i + 1, j)];
                    laplacian[(i, j)] =
                        (t_left + t_right + t_up + t_down - 4.0 * t_center) / (grid_x * grid_y);
                }
            }
            laplacian
        };

        return heat_src + self.deffusion_coeff * laplacian
            - self.decay_coeff * (self.temp.clone() - self.temp_room);
    }

    pub fn apply_boundary(&mut self) {
        let bw = 5 as usize;
        let (nx, ny) = self.temp.dim();

        for i in 0..nx {
            let t_start = self.temp_room;
            let t_end = self.temp[(i, bw)];
            for j in 0..bw {
                self.temp[(i, j)] = t_start + (t_end - t_start) * j as f64 / bw as f64;
            }
            let t_end = self.temp[(i, ny - 1 - bw)];
            for j in 0..bw {
                self.temp[(i, ny - 1 - j)] = t_start + (t_end - t_start) * j as f64 / bw as f64;
            }
        }
        for j in 0..ny {
            let t_start = self.temp_room;
            let t_end = self.temp[(bw, j)];
            for i in 0..bw {
                self.temp[(i, j)] = t_start + (t_end - t_start) * i as f64 / bw as f64;
            }
            let t_end = self.temp[(nx - 1 - bw, j)];
            for i in 0..bw {
                self.temp[(nx - 1 - i, j)] = t_start + (t_end - t_start) * i as f64 / bw as f64;
            }
        }
    }
}

pub struct SimulationConfig {
    penguin_max_vel: f64,
    penguin_radius: f64,
    heat_gen_coeff: f64,
    heat_p2e_coeff: f64,
    heat_e2p_coeff: f64,
}

impl SimulationConfig {
    pub fn new(
        penguin_max_vel: f64,
        penguin_radius: f64,
        heat_gen_coeff: f64,
        heat_p2e_coeff: f64,
        heat_e2p_coeff: f64,
    ) -> Self {
        Self {
            penguin_max_vel,
            penguin_radius,
            heat_gen_coeff,
            heat_p2e_coeff,
            heat_e2p_coeff,
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
            self.penguins
                .iter()
                .zip(t_airs.iter())
                .for_each(|(p, t_air)| {
                    let [x, y] = p.pos;
                    let (min_i, min_j) = self.air.pos2idx([x - 5. * heat_r, y - 5. * heat_r]);
                    let (max_i, max_j) = self.air.pos2idx([x + 5. * heat_r, y + 5. * heat_r]);
                    for i in min_i..max_i {
                        for j in min_j..max_j {
                            let [gx, gy] = self.air.idx2pos(i, j);
                            let dist2 = (gx - x) * (gx - x) + (gy - y) * (gy - y);
                            heat_src[(i, j)] += p2e_coeff
                                * (p.body_temp - t_air)
                                * (-0.5 * dist2 / (heat_r * heat_r)).exp();
                        }
                    }
                });
            self.air.get_derive(heat_src)
        };

        // compute velocities based on temperature gradient
        let vel_factor = self.config.penguin_max_vel / std::f64::consts::FRAC_PI_2;
        let velocities = self
            .penguins
            .iter()
            .enumerate()
            .map(|(_, p)| {
                let [gx, gy] = self.air.get_grad(p.pos);
                let grad_norm = (gx * gx + gy * gy).sqrt().max(1e-6);
                let dir = [-gx / grad_norm, -gy / grad_norm];
                let speed = vel_factor * (p.body_temp - p.prefer_temp).atan();
                [speed * dir[0], speed * dir[1]]
            })
            .collect::<Vec<_>>();

        (derive_body_temps, derive_air_temp, velocities)
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
            pos[0] = pos[0].clamp(0.0, self.air.size_x);
            pos[1] = pos[1].clamp(0.0, self.air.size_y);
        });

        // collision tree
        let kdtree = positions.iter().fold(
            KdTree::with_capacity(2, positions.len()),
            |mut kdtree, p| {
                kdtree.add(p, p).unwrap();
                kdtree
            },
        );

        // Resolve overlaps by pushing penguins apart
        let colli_dist = 2. * self.config.penguin_radius;
        let mut positions = positions
            .iter()
            .map(|pi| {
                let (count, acc_pos) = kdtree
                    .within(pi, colli_dist.powf(2.), &squared_euclidean)
                    .unwrap()
                    .into_iter()
                    .filter(|(dist2, _)| *dist2 > 0.0)
                    .map(|(dist2, pj)| {
                        let dx = pi[0] - pj[0];
                        let dy = pi[1] - pj[1];
                        let overlap = colli_dist - dist2.sqrt();
                        let corr_factor = 0.5 * overlap / colli_dist;
                        [corr_factor * dx, corr_factor * dy]
                    })
                    .fold((0u64, [0.0; 2]), |(i, acc), x| {
                        (i + 1, [acc[0] + x[0], acc[1] + x[1]])
                    });
                if count == 0 {
                    *pi
                } else {
                    [pi[0] + acc_pos[0], pi[1] + acc_pos[1]]
                }
            })
            .collect::<Vec<_>>();

        // Apply boundary conditions again
        positions.iter_mut().for_each(|pos| {
            pos[0] = pos[0].clamp(0.0, self.air.size_x);
            pos[1] = pos[1].clamp(0.0, self.air.size_y);
        });

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

        // Apply boundary conditions to air temperature after update
        self.air.apply_boundary();
    }
}
