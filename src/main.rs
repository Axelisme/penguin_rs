use ndarray::Array2;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal, Uniform};

use penguin_rs::simulator::{AirTemp, Penguin, Simulation, SimulationConfig};

fn main() {
    const N: usize = 100;
    const BOX_SIZE: f64 = 10.0;

    const SIM_TIME: f64 = 20.0;
    const DT: f64 = 0.001;

    const V0: f64 = 2.0;
    const COLL_R: f64 = 0.1;
    const GEN_COEFF: f64 = 4.0;
    const P2E_COEFF: f64 = 10.0;
    const E2P_COEFF: f64 = 0.1;
    const D_DIFFUSION: f64 = 0.4;
    const NATURE_DECAY: f64 = 5.0;
    const T_ROOM: f64 = -30.0;

    const T_BODY_MEAN: f64 = 19.0;
    const T_BODY_STD: f64 = 0.1;
    const T_PREFER_COMMON: f64 = 20.0;
    const M: usize = 120;
    const ENABLE_COLLISION: bool = true;

    let sim_config = SimulationConfig::new(
        V0,
        COLL_R,
        GEN_COEFF,
        P2E_COEFF,
        E2P_COEFF,
        ENABLE_COLLISION,
    );

    // init penguins
    let mut rng = StdRng::seed_from_u64(0);
    let pos_dist = Uniform::new(0.0, BOX_SIZE).unwrap();
    let norm = Normal::new(T_BODY_MEAN, T_BODY_STD).unwrap();
    let penguins = (0..N)
        .map(|_| {
            Penguin::new(
                [pos_dist.sample(&mut rng), pos_dist.sample(&mut rng)],
                [0.0, 0.0],
                norm.sample(&mut rng),
                T_PREFER_COMMON,
            )
        })
        .collect::<Vec<_>>();

    // init air grid
    let air_temp = AirTemp::new(
        Array2::from_elem((M, M), T_ROOM),
        BOX_SIZE,
        D_DIFFUSION,
        NATURE_DECAY,
        T_ROOM,
    );

    let total_steps = (SIM_TIME / DT) as usize;
    println!("Total steps: {}", total_steps);

    let mut sim = Simulation::new(sim_config, penguins, air_temp);
    for step in 0..total_steps {
        sim.step(DT);
        if step % 1000 == 0 {
            let max_body = sim
                .penguins
                .iter()
                .map(|p| p.body_temp)
                .fold(f64::MIN, f64::max);
            let min_body = sim
                .penguins
                .iter()
                .map(|p| p.body_temp)
                .fold(f64::MAX, f64::min);
            let max_air = sim.air.temp.iter().cloned().fold(f64::MIN, f64::max);
            println!(
                "Step {}: body_temp min={:.3}, max={:.3}, air_max={:.3}",
                step, min_body, max_body, max_air
            );
        }
    }
    println!("Simulation complete.");
}
