import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from penguin_rs import PySimulation

from util import (
    calculate_penguin_density_by_temp,
    calculate_temp_gradient_relationship,
    calculate_temperature_stats,
    create_penguin_colors,
    create_title_text,
    get_env_temps_at_positions,
    update_axis_limits,
    update_color_limits,
)

# Parameters based on main.rs
SEED = 1
NUM_PENGUINS = 500
PENGUIN_MOVE_FACTOR = 0.05
PENGUIN_RADIUS = 0.1
HEAT_GEN_COEFF = 0.15
HEAT_P2E_COEFF = 1.0
HEAT_E2P_COEFF = 0.01
PREFER_TEMP_COMMON = 20.0
INIT_TEMP_MEAN = PREFER_TEMP_COMMON
NUM_GRID = 180
BOX_SIZE = 9.0
DEFFUSION_COEFF = 0.4
DECAY_COEFF = 0.4
TEMP_ROOM = -30.0
ENABLE_COLLISION = True


DESITY_FACTOR = 2.0
init_penguin_positions = (
    (np.random.rand(NUM_PENGUINS, 2) - 0.5)
    * DESITY_FACTOR
    * np.sqrt(NUM_PENGUINS)
    * PENGUIN_RADIUS
) + BOX_SIZE / 2
init_penguin_temps = np.full(NUM_PENGUINS, INIT_TEMP_MEAN)
init_air_temp = np.full((NUM_GRID, NUM_GRID), 0.3 * TEMP_ROOM + 0.7 * INIT_TEMP_MEAN)

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
    prefer_temp_common=PREFER_TEMP_COMMON,
    box_size=BOX_SIZE,
    deffusion_coeff=DEFFUSION_COEFF,
    decay_coeff=DECAY_COEFF,
    temp_room=TEMP_ROOM,
    enable_collision=ENABLE_COLLISION,
)


# --- Plotting Parameters ---
SIM_TIME = 100.0
DT = 0.001
TOTAL_STEPS = int(SIM_TIME / DT)
FRAMES_PER_SECOND = 10  # Target FPS for animation
# Simulation steps per animation frame
STEPS_PER_FRAME = max(1, int(1 / (FRAMES_PER_SECOND * DT)))
TOTAL_FRAMES = int(TOTAL_STEPS / STEPS_PER_FRAME)

print(f"Initializing simulation with {NUM_PENGUINS} penguins...")
print(f"Grid size: {NUM_GRID}x{NUM_GRID}, Box size: {BOX_SIZE}")
print(
    f"Simulation time: {SIM_TIME}s, dt: {DT}, Tcalculate_penguin_density_by_tempotal steps: {TOTAL_STEPS}"
)
print(
    f"Animation: Target FPS: {FRAMES_PER_SECOND}, Steps/Frame: {STEPS_PER_FRAME}, Total Frames: {TOTAL_FRAMES}"
)

# --- Matplotlib Setup ---
fig = plt.figure(figsize=(16, 12))
ax_scatter = plt.subplot(2, 2, 1)  # Top left
ax_main = plt.subplot(2, 2, 2)  # Top right
ax_density = plt.subplot(2, 2, 3)  # Bottom left - NEW: Temperature vs Penguin Density
ax_gradient = plt.subplot(2, 2, 4)  # Bottom right

# Initial state for plotting setup
positions_init, _, body_temps_init, air_temps_vec_init = sim.get_state()
air_temp_grid_init = np.array(air_temps_vec_init).reshape((NUM_GRID, NUM_GRID))


# Calculate initial temperature-gradient relationship
temp_range_init, avg_gradients_init = calculate_temp_gradient_relationship(
    air_temp_grid_init
)

# Calculate initial penguin density by temperature
temp_range_density_init, densities_init = calculate_penguin_density_by_temp(
    positions_init, air_temp_grid_init, BOX_SIZE, NUM_GRID
)
densities_init *= 2 * np.sqrt(3) * PENGUIN_RADIUS * PENGUIN_RADIUS

# Calculate initial environmental temperatures at penguin positions
env_temps_init = get_env_temps_at_positions(
    positions_init, air_temp_grid_init, BOX_SIZE, NUM_GRID
)

# Create binary colors based on temperature preference
penguin_colors_init = create_penguin_colors(body_temps_init, PREFER_TEMP_COMMON)

# Left subplot: Body temperature vs Environmental temperature scatter plot
scatter_temp = ax_scatter.scatter(
    body_temps_init,
    env_temps_init,
    edgecolor="gray",
    s=10,
    alpha=0.7,
    animated=True,
)
# ax_scatter.set_xlim(PREFER_TEMP_COMMON - 1, PREFER_TEMP_COMMON + 1)
ax_scatter.set_ylim(TEMP_ROOM + 10, PREFER_TEMP_COMMON - 5)

# Right subplot: Main simulation view
im = ax_main.imshow(
    air_temp_grid_init.T,
    cmap="coolwarm",
    origin="lower",
    extent=[0, BOX_SIZE, 0, BOX_SIZE],
    interpolation="nearest",
    animated=True,
)
scatter_main = ax_main.scatter(
    [p[0] for p in positions_init],
    [p[1] for p in positions_init],
    c=penguin_colors_init,
    edgecolor="none",
    s=5,
    animated=True,
)

# Bottom right subplot: Temperature vs Gradient relationship
(gradient_line,) = ax_gradient.plot(
    temp_range_init, avg_gradients_init, "b-", linewidth=2, animated=True
)

# Bottom left subplot: Temperature vs Penguin Density
(density_line,) = ax_density.plot(
    temp_range_density_init, densities_init, "g-", linewidth=2, animated=True
)

# Plot settings
ax_scatter.set_xlabel("Body Temperature (°C)")
ax_scatter.set_ylabel("Environmental Temperature (°C)")
ax_scatter.set_title("Body vs Environmental Temperature")
ax_scatter.grid(True, alpha=0.3)
# Add reference line for preferred temperature
ax_scatter.axvline(
    x=PREFER_TEMP_COMMON,
    color="red",
    linestyle="--",
    alpha=0.7,
    label=f"Preferred Temp ({PREFER_TEMP_COMMON}°C)",
)
ax_scatter.legend()

ax_main.set_xlim(0, BOX_SIZE)
ax_main.set_ylim(0, BOX_SIZE)
ax_main.set_xlabel("X position")
ax_main.set_ylabel("Y position")
title = ax_main.set_title("Penguin Simulation - Step 0")
ax_main.set_aspect("equal", adjustable="box")

# Gradient plot settings
ax_gradient.set_xlabel("Temperature (°C)")
ax_gradient.set_ylabel("Average Gradient")
ax_gradient.set_title("Temperature vs Gradient Relationship")
ax_gradient.grid(True, alpha=0.3)

# Density plot settings
ax_density.set_xlabel("Temperature (°C)")
ax_density.set_ylabel("Penguin Density (penguins/unit²)")
ax_density.set_title("Temperature vs Penguin Density")
ax_density.grid(True, alpha=0.3)

# Add colorbar for main plot
fig.colorbar(im, ax=ax_main, label="Air Temperature (°C)")


# --- Animation Update Function ---
def update(frame):
    global sim
    start_frame_time = time.time()

    # Run simulation steps for this frame
    for _ in range(STEPS_PER_FRAME):
        sim.step(DT)

    # Get current state
    positions, _, body_temps, air_temps_vec = sim.get_state()

    pos_array = np.array(positions)
    body_temps = np.array(body_temps)
    air_temp_grid = np.array(air_temps_vec).reshape((NUM_GRID, NUM_GRID))

    # Create binary colors based on temperature preference
    penguin_colors = create_penguin_colors(body_temps, PREFER_TEMP_COMMON)

    # Calculate environmental temperatures at current penguin positions
    env_temps = get_env_temps_at_positions(positions, air_temp_grid, BOX_SIZE, NUM_GRID)

    # --- Update Plot Data ---
    # Update main simulation view
    im.set_data(air_temp_grid.T)

    scatter_main.set_offsets(pos_array)
    scatter_main.set_color(penguin_colors)

    # Update temperature scatter plot
    scatter_temp.set_offsets(np.column_stack([body_temps, env_temps]))

    # Update gradient plot
    temp_range, avg_gradients = calculate_temp_gradient_relationship(air_temp_grid)
    gradient_line.set_data(temp_range, avg_gradients)

    # Update density plot
    temp_range_density, densities = calculate_penguin_density_by_temp(
        positions, air_temp_grid, BOX_SIZE, NUM_GRID
    )
    densities *= 2 * np.sqrt(3) * PENGUIN_RADIUS * PENGUIN_RADIUS
    density_line.set_data(temp_range_density, densities)

    # Update gradient plot limits
    if len(avg_gradients) > 0:
        ax_gradient.set_xlim(np.min(temp_range), np.max(temp_range))
        update_axis_limits(ax_gradient, avg_gradients)

    # Update density plot limits
    if len(densities) > 0:
        ax_density.set_xlim(np.min(temp_range_density), np.max(temp_range_density))
        update_axis_limits(ax_density, densities)

    # --- Update Color Limits Dynamically ---
    air_stats, body_stats = calculate_temperature_stats(air_temp_grid, body_temps)
    air_min, air_mid, air_max = air_stats
    body_min, body_mid, body_max = body_stats

    ax_scatter.set_xlim(body_min - 1, body_max + 1)
    ax_scatter.set_ylim(air_min - 1, air_max + 1)

    # --- Update Title and Print Status ---
    current_sim_time = (frame + 1) * STEPS_PER_FRAME * DT
    frame_time = time.time() - start_frame_time

    # Create title text with frame time included
    base_title = create_title_text(
        current_sim_time, frame, TOTAL_FRAMES, air_stats, body_stats
    )
    title_text = f"{base_title} (Frame time: {frame_time * 1000:.1f}ms)"
    title.set_text(title_text)

    update_color_limits(im, air_temp_grid)

    print(f"\r{title_text}", end="\r")

    return im, scatter_main, scatter_temp, gradient_line, density_line, title


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
#     f"penguin_simulation_colli_{ENABLE_COLLISION}.mp4",
#     writer="ffmpeg",
#     fps=FRAMES_PER_SECOND,
# )
plt.show()

end_time = time.time()
print(f"\nSimulation and animation complete. Total time: {end_time - start_time:.2f}s")
