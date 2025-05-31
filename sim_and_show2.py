import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from penguin_rs import PySimulation

from util import (
    calculate_penguin_density_by_temp,
    calculate_stable_temp,
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
DIFFUSION_COEFF = 0.4
DECAY_COEFF = 0.4
TEMP_ROOM = -30.0
COLLISION_STRENGTH = 10.0  # 碰撞排斥力强度

print(
    calculate_stable_temp(
        HEAT_GEN_COEFF,
        HEAT_E2P_COEFF,
        DIFFUSION_COEFF,
        HEAT_P2E_COEFF,
        PENGUIN_RADIUS,
        DECAY_COEFF,
        TEMP_ROOM,
    )
)
# exit()

# Parameter mapping for theoretical model
A = HEAT_GEN_COEFF
B = HEAT_E2P_COEFF
x0 = PREFER_TEMP_COMMON
alpha = PENGUIN_MOVE_FACTOR


# Theoretical functions from evolution_system.py
def grad_func(y):
    """Theoretical gradient function"""
    peak_y = -3
    return 1.6 * np.where(
        y < peak_y, (y + 30) / (peak_y + 30), (10 - y) / (10 - peak_y)
    )


def density_factor(y):
    """Theoretical density factor function"""
    return 0.5 + np.arctan((y + 3) / 0.7) / np.pi
    # return np.zeros_like(y)


def H(density, grad):
    return -alpha * (1.0 - density) * grad**2
    # return -alpha * grad**2


def H_predict(x, y):
    return H(density_factor(y), grad_func(y))


def theoretical_vector_field(x, y):
    """Calculate theoretical dx/dt and dy/dt"""
    dxdt = A - B * (x - y)
    dydt = (x - x0) * H_predict(x, y)
    return dxdt, dydt


def actual_vector_field(x, y, grad_temps, gradients, density_temps, densities):
    actual_grad = np.interp(y, grad_temps, gradients, left=0, right=0)
    actual_density = np.interp(y, density_temps, densities, left=0, right=0)

    dxdt = A - B * (x - y)
    dydt = (x - x0) * H(actual_density, actual_grad)

    return dxdt, dydt


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
    diffusion_coeff=DIFFUSION_COEFF,
    decay_coeff=DECAY_COEFF,
    temp_room=TEMP_ROOM,
    collision_strength=COLLISION_STRENGTH,
)

# --- Plotting Parameters ---
SIM_TIME = 100.0
DT = 0.003
TOTAL_STEPS = int(SIM_TIME / DT)
FRAMES_PER_SECOND = 10  # Target FPS for animation
STEPS_PER_FRAME = max(1, int(1 / (FRAMES_PER_SECOND * DT)))
TOTAL_FRAMES = int(TOTAL_STEPS / STEPS_PER_FRAME)

print(f"Initializing simulation with {NUM_PENGUINS} penguins...")
print(f"Grid size: {NUM_GRID}x{NUM_GRID}, Box size: {BOX_SIZE}")
print(f"Simulation time: {SIM_TIME}s, dt: {DT}, Total steps: {TOTAL_STEPS}")
print(
    f"Animation: Target FPS: {FRAMES_PER_SECOND}, Steps/Frame: {STEPS_PER_FRAME}, Total Frames: {TOTAL_FRAMES}"
)


# Initial state for plotting setup
positions, _, body_temps, air_temps = sim.get_state()
positions = np.array(positions)
body_temps = np.array(body_temps)
air_temps = np.array(air_temps).reshape((NUM_GRID, NUM_GRID))

# Calculate initial temperature-gradient relationship
grad_temps, gradients = calculate_temp_gradient_relationship(air_temps)

# Calculate initial penguin density by temperature
density_temps, densities = calculate_penguin_density_by_temp(
    positions, air_temps, BOX_SIZE, NUM_GRID
)
densities *= 2 * np.sqrt(3) * PENGUIN_RADIUS * PENGUIN_RADIUS
densities = np.clip(densities, 0, 1)

# Calculate initial environmental temperatures at penguin positions
env_temps = get_env_temps_at_positions(positions, air_temps, BOX_SIZE, NUM_GRID)

# Create binary colors based on temperature preference
penguin_colors = create_penguin_colors(body_temps, PREFER_TEMP_COMMON)

# Calculate theoretical relationships
pred_gradients = grad_func(grad_temps)
pred_densities = density_factor(density_temps)


# --- Matplotlib Setup ---
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
ax_scatter, ax_main = axs[0]
ax_density, ax_gradient = axs[1]

# Left subplot: Body temperature vs Environmental temperature scatter plot with vector field
scatter_temp = ax_scatter.scatter(
    body_temps, env_temps, edgecolor="gray", s=10, alpha=0.7, animated=True
)


# Initialize vector field grid based on initial data range
x_range = [body_temps.min() - 2, body_temps.max() + 2]
y_range = [env_temps.min() - 2, env_temps.max() + 2]
x_vec = np.linspace(x_range[0], x_range[1], 100)
y_vec = np.linspace(y_range[0], y_range[1], 100)
Xs, Ys = np.meshgrid(x_vec, y_vec)
U_vec, V_vec = actual_vector_field(
    Xs, Ys, grad_temps, gradients, density_temps, densities
)

# Plot vector field (this will be recreated each frame)
quiver = ax_scatter.quiver(
    Xs[::10, ::10],
    Ys[::10, ::10],
    U_vec[::10, ::10],
    V_vec[::10, ::10],
    animated=True,
    scale=10,
    width=0.002,
)


# contour of dx/dt = 0 and dy/dt = 0
dx_contour = ax_scatter.contour(
    Xs, Ys, U_vec, levels=[0], colors="red", linestyles="--", alpha=0.8, linewidths=2
)
dy_contour = ax_scatter.axvline(PREFER_TEMP_COMMON, color="blue", alpha=0.8)


# Right subplot: Main simulation view
im = ax_main.imshow(
    air_temps.T,
    cmap="coolwarm",
    origin="lower",
    extent=[0, BOX_SIZE, 0, BOX_SIZE],
    interpolation="none",
    animated=True,
)
scatter_main = ax_main.scatter(
    positions[:, 0],
    positions[:, 1],
    c=penguin_colors,
    edgecolor="none",
    animated=True,
    s=10,
)

# Bottom right subplot: Temperature vs Gradient relationship
(gradient_pred_line,) = ax_gradient.plot(
    grad_temps, pred_gradients, label="Theoretical", animated=True
)
(gradient_line,) = ax_gradient.plot(
    grad_temps, gradients, animated=True, label="Actual"
)

# Bottom left subplot: Temperature vs Penguin Density
(density_pred_line,) = ax_density.plot(
    density_temps, pred_densities, label="Theoretical", animated=True
)
(density_line,) = ax_density.plot(
    density_temps, densities, animated=True, label="Actual"
)


# Plot settings
ax_scatter.set_xlabel("Body Temperature (°C)")
ax_scatter.set_ylabel("Environmental Temperature (°C)")
ax_scatter.set_title("Body vs Env Temp + Dynamic Vector Field + Isoclines")

ax_main.set_xlim(0, BOX_SIZE)
ax_main.set_ylim(0, BOX_SIZE)
ax_main.set_xlabel("X position")
ax_main.set_ylabel("Y position")
title = ax_main.set_title("Penguin Simulation - Step 0")
ax_main.set_aspect("equal", adjustable="box")

# Gradient plot settings
ax_gradient.set_xlim(TEMP_ROOM, PREFER_TEMP_COMMON)
ax_gradient.set_ylim(0, 2.5)
ax_gradient.set_xlabel("Temperature (°C)")
ax_gradient.set_ylabel("Average Gradient")
ax_gradient.set_title("Temperature vs Gradient Relationship")
ax_gradient.legend()

# Density plot settings
ax_density.set_xlim(TEMP_ROOM, PREFER_TEMP_COMMON)
ax_density.set_ylim(0, 1.1)
ax_density.set_xlabel("Temperature (°C)")
ax_density.set_ylabel("Penguin Density (penguins/unit²)")
ax_density.set_title("Temperature vs Penguin Density")
ax_density.legend()


# --- Animation Update Function ---
def update(frame):
    start_frame_time = time.time()

    # Run simulation steps for this frame
    for _ in range(STEPS_PER_FRAME):
        sim.step(DT)

    # Get current state
    positions, _, body_temps, air_temps = sim.get_state()
    positions = np.array(positions)
    body_temps = np.array(body_temps)
    air_temps = np.array(air_temps).reshape((NUM_GRID, NUM_GRID))

    # Create binary colors based on temperature preference
    penguin_colors = create_penguin_colors(body_temps, PREFER_TEMP_COMMON)

    # Calculate environmental temperatures at current penguin positions
    env_temps = get_env_temps_at_positions(positions, air_temps, BOX_SIZE, NUM_GRID)

    # --- Update Plot Data ---
    # Update main simulation view
    im.set_data(air_temps.T)

    scatter_main.set_offsets(positions)
    scatter_main.set_color(penguin_colors)

    # Update temperature scatter plot
    scatter_temp.set_offsets(np.column_stack([body_temps, env_temps]))

    # Update gradient plot
    grad_temps, gradients = calculate_temp_gradient_relationship(air_temps)
    gradient_line.set_data(grad_temps, gradients)

    # Update density plot
    density_temps, densities = calculate_penguin_density_by_temp(
        positions, air_temps, BOX_SIZE, NUM_GRID
    )
    densities *= 2 * np.sqrt(3) * PENGUIN_RADIUS * PENGUIN_RADIUS
    densities = np.clip(densities, 0, 1)
    density_line.set_data(density_temps, densities)

    # Update gradient plot limits
    pred_gradients = grad_func(grad_temps)
    pred_densities = density_factor(density_temps)
    gradient_pred_line.set_data(grad_temps, pred_gradients)
    density_pred_line.set_data(density_temps, pred_densities)

    update_axis_limits(ax_gradient, np.concatenate([gradients, pred_gradients]))
    update_axis_limits(ax_density, np.concatenate([densities, pred_densities]))

    # Dynamically update vector field grid based on current data range
    x_range = [np.nanmin(body_temps) - 2, np.nanmax(body_temps) + 2]
    y_range = [np.nanmin(env_temps) - 2, np.nanmax(env_temps) + 2]

    # update scatter plot limits
    ax_scatter.set_xlim(*x_range)
    ax_scatter.set_ylim(*y_range)

    # Update vector field using actual computed gradients and densities
    x_vec = np.linspace(x_range[0], x_range[1], 100)
    y_vec = np.linspace(y_range[0], y_range[1], 100)
    Xs, Ys = np.meshgrid(x_vec, y_vec)
    U_vec, V_vec = actual_vector_field(
        Xs, Ys, grad_temps, gradients, density_temps, densities
    )

    # Update quiver positions and vectors
    global quiver
    quiver.remove()
    quiver = ax_scatter.quiver(
        Xs[::10, ::10],
        Ys[::10, ::10],
        U_vec[::10, ::10],
        V_vec[::10, ::10],
        animated=True,
        scale=10,
        width=0.002,
    )

    # update contour of dx/dt = 0 and dy/dt = 0

    # contour of dx/dt = 0 and dy/dt = 0
    global dx_contour, dy_contour
    dx_contour.remove()
    dy_contour.remove()
    dx_contour = ax_scatter.contour(
        Xs,
        Ys,
        U_vec,
        levels=[0],
        colors="red",
        linestyles="--",
        alpha=0.8,
        linewidths=2,
    )
    dy_contour = ax_scatter.axvline(PREFER_TEMP_COMMON, color="blue", alpha=0.8)

    # Adjust quiver scale dynamically based on vector magnitude
    max_magnitude = np.sqrt(U_vec**2 + V_vec**2).max()
    if max_magnitude > 0:
        quiver.scale = max_magnitude * 10

    # --- Update Color Limits Dynamically ---
    air_stats, body_stats = calculate_temperature_stats(air_temps, body_temps)

    # --- Update Title and Print Status ---
    current_sim_time = (frame + 1) * STEPS_PER_FRAME * DT
    frame_time = time.time() - start_frame_time

    # Create title text with frame time included
    base_title = create_title_text(
        current_sim_time, frame, TOTAL_FRAMES, air_stats, body_stats
    )
    title_text = f"{base_title} (Frame time: {frame_time * 1000:.1f}ms)"
    title.set_text(title_text)

    update_color_limits(im, air_temps)

    print(f"\r{title_text}", end="\r")

    return (
        im,
        scatter_main,
        scatter_temp,
        quiver,
        gradient_line,
        density_line,
        gradient_pred_line,
        density_pred_line,
        title,
        dx_contour,
        dy_contour,
    )


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
