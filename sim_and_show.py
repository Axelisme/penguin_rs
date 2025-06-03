import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from penguin_rs import PySimulation

from util import calculate_temp_gradient_relationship, get_env_temps_at_positions
from util.animation import GradPlot, PenguinPlot, PhasePlot, VelocityRatioPlot2

# Parameters based on main.rs
SEED = 1
NUM_PENGUINS = 500
PENGUIN_MOVE_FACTOR = 0.05
PENGUIN_RADIUS = 0.1
HEAT_GEN_COEFF = 0.15
HEAT_P2E_COEFF = 1.0
HEAT_E2P_COEFF = 0.01
PREFER_TEMP = 20.0
INIT_TEMP_MEAN = PREFER_TEMP
NUM_GRID = 180
BOX_SIZE = 9.0
DIFFUSION_COEFF = 0.4
DECAY_COEFF = 0.4
TEMP_ROOM = -30.0
COLLISION_STRENGTH = 10.0  # 碰撞排斥力强度


DENSITY_FACTOR = 2.0
init_penguin_positions = (
    (np.random.rand(NUM_PENGUINS, 2) - 0.5)
    * DENSITY_FACTOR
    * np.sqrt(NUM_PENGUINS)
    * PENGUIN_RADIUS
) + BOX_SIZE / 2
init_penguin_temps = np.full(NUM_PENGUINS, INIT_TEMP_MEAN)
init_air_temp = np.full((NUM_GRID, NUM_GRID), 0.2 * TEMP_ROOM + 0.8 * INIT_TEMP_MEAN)

init_penguin_infos = np.concatenate(
    [init_penguin_positions, init_penguin_temps[:, None]], axis=1
)

# Create the simulation instance
sim = PySimulation(
    init_penguins=init_penguin_infos,
    init_air_temp=init_air_temp,
    penguin_move_factor=PENGUIN_MOVE_FACTOR,
    penguin_radius=PENGUIN_RADIUS,
    heat_gen_coeff=HEAT_GEN_COEFF,
    heat_p2e_coeff=HEAT_P2E_COEFF,
    heat_e2p_coeff=HEAT_E2P_COEFF,
    prefer_temp=PREFER_TEMP,
    box_size=BOX_SIZE,
    diffusion_coeff=DIFFUSION_COEFF,
    decay_coeff=DECAY_COEFF,
    temp_room=TEMP_ROOM,
    collision_strength=COLLISION_STRENGTH,
)

# --- Plotting Parameters ---
SIM_TIME = 1000.0
DT = 0.003
TOTAL_STEPS = int(SIM_TIME / DT)
FRAMES_PER_SECOND = 20  # Target FPS for animation
STEPS_PER_FRAME = max(1, int(1 / (FRAMES_PER_SECOND * DT)))
TOTAL_FRAMES = int(TOTAL_STEPS / STEPS_PER_FRAME)

# Note: GradPlot improvements made:
# 1. Shows temperature vs gradient relationship with actual data
# 2. Includes Gaussian function fitting for better trend visualization
# 3. Updates in real-time to show how the relationship evolves during simulation

print(f"Initializing simulation with {NUM_PENGUINS} penguins...")
print(f"Grid size: {NUM_GRID}x{NUM_GRID}, Box size: {BOX_SIZE}")
print(f"Simulation time: {SIM_TIME}s, dt: {DT}, Total steps: {TOTAL_STEPS}")
print(
    f"Animation: Target FPS: {FRAMES_PER_SECOND}, Steps/Frame: {STEPS_PER_FRAME}, Total Frames: {TOTAL_FRAMES}"
)


# Initial state for plotting setup
positions, velocities, body_temps, air_temps = sim.get_state()


# Calculate initial temperature-gradient relationship
grad_temps, gradients = calculate_temp_gradient_relationship(air_temps, BOX_SIZE)


def make_title(current_t, frame, total_frames, air_temps, body_temps):
    air_stats = (np.min(air_temps), np.median(air_temps), np.max(air_temps))
    body_stats = (np.min(body_temps), np.median(body_temps), np.max(body_temps))
    return (
        f"Penguin Sim - Time: {current_t:.2f}s Frame: {frame}/{total_frames} "
        f"Body T: [{body_stats[0]:.2f}, {body_stats[1]:.2f},{body_stats[2]:.2f}] "
        f"Air T: [{air_stats[0]:.2f}, {air_stats[1]:.2f},{air_stats[2]:.2f}]"
    )


# --- Matplotlib Setup ---
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
ax_scatter, ax_main = axs[0]
ax_density, ax_gradient = axs[1]


# Initialize plot objects
# Calculate environmental temperatures at penguin positions for PhasePlot
env_temps = get_env_temps_at_positions(positions, air_temps, BOX_SIZE)
phase_plot = PhasePlot(
    ax_scatter,
    body_temps,
    env_temps,
    PREFER_TEMP,
    HEAT_GEN_COEFF,
    HEAT_E2P_COEFF,
    TEMP_ROOM,
)

penguin_plot = PenguinPlot(
    ax_main, positions, body_temps, air_temps, BOX_SIZE, PREFER_TEMP, TOTAL_FRAMES
)

penguin_gradients = np.interp(env_temps, grad_temps, gradients)

vel_ratio_plot = VelocityRatioPlot2(
    ax_density,
    body_temps,
    env_temps,
    penguin_gradients,
    PENGUIN_MOVE_FACTOR,
    PREFER_TEMP,
    TEMP_ROOM,
    DT * STEPS_PER_FRAME,
)

# Initialize gradient plot
grad_plot = GradPlot(
    ax_gradient,
    grad_temps,
    gradients,
    TEMP_ROOM,
    PREFER_TEMP,
)


# --- Animation Update Function ---
def update(frame):
    # Run simulation steps
    for step in range(STEPS_PER_FRAME):
        sim.step(DT)

    current_t = (frame + 1) * STEPS_PER_FRAME * DT

    # Get final state for other plots (updated once per frame)
    positions, _, body_temps, air_temps = sim.get_state()

    # Calculate temperature gradient relationship
    grad_temps, gradients = calculate_temp_gradient_relationship(air_temps, BOX_SIZE)

    # Calculate environmental temperatures at penguin positions
    env_temps = get_env_temps_at_positions(positions, air_temps, BOX_SIZE)
    penguin_gradients = np.interp(env_temps, grad_temps, gradients)

    # Update other plots (once per frame for better performance)
    phase_artists = phase_plot.update(body_temps, env_temps)
    penguin_artists = penguin_plot.update(
        positions, body_temps, air_temps, current_t, frame
    )
    vel_ratio_artists = vel_ratio_plot.update(body_temps, env_temps, penguin_gradients)
    grad_artists = grad_plot.update(grad_temps, gradients)

    title_text = make_title(current_t, frame, TOTAL_FRAMES, air_temps, body_temps)

    print(f"\r{title_text}", end="\r")

    # Return all artists for blitting
    return phase_artists + penguin_artists + vel_ratio_artists + grad_artists


# --- Save Animation as GIF ---
print("Rendering animation to GIF (this may take a while)...")
start_time = time.time()

ani = animation.FuncAnimation(
    fig,
    update,
    frames=TOTAL_FRAMES,
    interval=1000 / FRAMES_PER_SECOND,
    blit=True,
    repeat=False,
)

plt.tight_layout()
# ani.save(
#     f"penguin_simulation_theory_colli_{ENABLE_COLLISION}.mp4",
#     writer="ffmpeg",
#     fps=FRAMES_PER_SECOND,
# )
plt.show()

end_time = time.time()
print(f"\nSimulation and animation complete. Total time: {end_time - start_time:.2f}s")
