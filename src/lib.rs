use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use simulator::{AirTemp, Simulation, SimulationConfig};

pub mod simulator;

#[pyclass]
struct PySimulation {
    simulation: Simulation,
}

#[pymethods]
impl PySimulation {
    #[new]
    fn new(
        init_penguins: PyReadonlyArray2<f64>,
        init_air_temp: PyReadonlyArray2<f64>,
        penguin_move_factor: f64,
        penguin_radius: f64,
        heat_gen_coeff: f64,
        heat_p2e_coeff: f64,
        heat_e2p_coeff: f64,
        prefer_temp_common: f64,
        box_size: f64,
        diffusion_coeff: f64,
        decay_coeff: f64,
        temp_room: f64,
        collision_strength: f64,
    ) -> PyResult<Self> {
        let config = SimulationConfig::new(
            penguin_move_factor,
            penguin_radius,
            heat_gen_coeff,
            heat_p2e_coeff,
            heat_e2p_coeff,
            collision_strength,
        );

        let init_penguins_arr = init_penguins.as_array();
        if init_penguins_arr.ndim() != 2 || init_penguins_arr.shape()[1] != 3 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "init_penguins 必須是一個形狀為 (num_penguins, 3) 的二維陣列".to_string(),
            ));
        }

        let num_penguins = init_penguins_arr.shape()[0];
        let mut positions = Vec::with_capacity(num_penguins);
        let mut body_temps = Vec::with_capacity(num_penguins);

        for row in init_penguins_arr.rows() {
            positions.push([row[0], row[1]]);
            body_temps.push(row[2]);
        }

        let velocities = vec![[0.0, 0.0]; num_penguins];
        let prefer_temps = vec![prefer_temp_common; num_penguins];

        let init_air_temp_arr = init_air_temp.as_array();
        if init_air_temp_arr.ndim() != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "init_air_temp 必須是一個二維陣列".to_string(),
            ));
        }
        if init_air_temp_arr.shape()[0] != init_air_temp_arr.shape()[1] {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "init_air_temp 必須是一個方形陣列 (num_grid, num_grid)".to_string(),
            ));
        }

        let air_temp = AirTemp::new(
            init_air_temp_arr.to_owned(),
            box_size,
            diffusion_coeff,
            decay_coeff,
            temp_room,
        );

        let simulation = Simulation::new(
            config,
            positions,
            velocities,
            body_temps,
            prefer_temps,
            air_temp,
        );
        Ok(Self { simulation })
    }

    fn get_state(&self) -> PyResult<(Vec<[f64; 2]>, Vec<[f64; 2]>, Vec<f64>, Vec<Vec<f64>>)> {
        let positions = self.simulation.positions.clone();
        let velocities = self.simulation.velocities.clone();
        let body_temps = self.simulation.body_temps.clone();
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
