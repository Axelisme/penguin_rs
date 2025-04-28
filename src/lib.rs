use ndarray::Array2;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Uniform};
use simulator::{AirTemp, Penguin, Simulation, SimulationConfig};

pub mod simulator;

#[pyclass]
struct PySimulation {
    simulation: Simulation,
}

#[pymethods]
impl PySimulation {
    #[new]
    fn new(
        seed: u64,
        num_penguins: usize,
        penguin_max_vel: f64,
        penguin_radius: f64,
        heat_gen_coeff: f64,
        heat_p2e_coeff: f64,
        heat_e2p_coeff: f64,
        init_temp_mean: f64,
        init_temp_std: f64,
        prefer_temp_common: f64,
        num_grid: usize,
        box_size: f64,
        deffusion_coeff: f64,
        decay_coeff: f64,
        temp_room: f64,
    ) -> PyResult<Self> {
        let config = SimulationConfig::new(
            penguin_max_vel,
            penguin_radius,
            heat_gen_coeff,
            heat_p2e_coeff,
            heat_e2p_coeff,
        );

        let mut rng = StdRng::seed_from_u64(seed);
        let pos_dist = Uniform::new(0.2 * box_size, 0.8 * box_size).unwrap();
        let temp_dist = Normal::new(init_temp_mean, init_temp_std).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid temp distribution: {}",
                e
            ))
        })?;

        let penguins = (0..num_penguins)
            .map(|_| {
                Penguin::new(
                    [pos_dist.sample(&mut rng), pos_dist.sample(&mut rng)],
                    [0.0, 0.0],
                    temp_dist.sample(&mut rng),
                    prefer_temp_common,
                )
            })
            .collect::<Vec<_>>();

        let air_temp = AirTemp::new(
            Array2::from_elem((num_grid, num_grid), temp_room),
            box_size,
            deffusion_coeff,
            decay_coeff,
            temp_room,
        );

        let simulation = Simulation::new(config, penguins, air_temp);
        Ok(Self { simulation })
    }

    fn get_state(&self) -> PyResult<(Vec<[f64; 2]>, Vec<[f64; 2]>, Vec<f64>, Vec<Vec<f64>>)> {
        let positions = self.simulation.penguins.iter().map(|p| p.pos).collect();
        let velocities = self.simulation.penguins.iter().map(|p| p.vel).collect();
        let body_temps = self
            .simulation
            .penguins
            .iter()
            .map(|p| p.body_temp)
            .collect();
        let air_temps_vec = self
            .simulation
            .air
            .temp
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect();
        Ok((positions, velocities, body_temps, air_temps_vec))
    }

    fn step(&mut self, dt: f64) -> PyResult<()> {
        self.simulation.step(dt);
        Ok(())
    }
}

#[pymodule]
fn penguin_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySimulation>()?;
    Ok(())
}
