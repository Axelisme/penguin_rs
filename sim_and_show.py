import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from penguin_rs import PySimulation

# Parameters based on main.rs
SEED = 1
NUM_PENGUINS = 500
PENGUIN_MAX_VEL = 2.0
PENGUIN_RADIUS = 0.1
HEAT_GEN_COEFF = 1.5
HEAT_P2E_COEFF = 10.0
HEAT_E2P_COEFF = 0.1
INIT_TEMP_MEAN = 19.0
INIT_TEMP_STD = 0.1
PREFER_TEMP_COMMON = 20.0
NUM_GRID = 140
BOX_SIZE = 10.0
DEFFUSION_COEFF = 0.4
DECAY_COEFF = 4.0
TEMP_ROOM = -30.0


# Create the simulation instance
sim = PySimulation(
    seed=SEED,
    num_penguins=NUM_PENGUINS,
    penguin_max_vel=PENGUIN_MAX_VEL,
    penguin_radius=PENGUIN_RADIUS,
    heat_gen_coeff=HEAT_GEN_COEFF,
    heat_p2e_coeff=HEAT_P2E_COEFF,
    heat_e2p_coeff=HEAT_E2P_COEFF,
    init_temp_mean=INIT_TEMP_MEAN,
    init_temp_std=INIT_TEMP_STD,
    prefer_temp_common=PREFER_TEMP_COMMON,
    num_grid=NUM_GRID,
    box_size=BOX_SIZE,
    deffusion_coeff=DEFFUSION_COEFF,
    decay_coeff=DECAY_COEFF,
    temp_room=TEMP_ROOM,
)


# --- Plotting Parameters ---
SIM_TIME = 500.0
DT = 0.001
TOTAL_STEPS = int(SIM_TIME / DT)
FRAMES_PER_SECOND = 30  # Target FPS for animation
# Simulation steps per animation frame
STEPS_PER_FRAME = max(1, int(1 / (FRAMES_PER_SECOND * DT)))
TOTAL_FRAMES = int(TOTAL_STEPS / STEPS_PER_FRAME)

print(f"Initializing simulation with {NUM_PENGUINS} penguins...")
print(f"Grid size: {NUM_GRID}x{NUM_GRID}, Box size: {BOX_SIZE}")
print(f"Simulation time: {SIM_TIME}s, dt: {DT}, Total steps: {TOTAL_STEPS}")
print(
    f"Animation: Target FPS: {FRAMES_PER_SECOND}, Steps/Frame: {STEPS_PER_FRAME}, Total Frames: {TOTAL_FRAMES}"
)

# --- Matplotlib Setup ---
fig, ax = plt.subplots(figsize=(8, 8))

# Initial state for plotting setup
positions_init, velocities_init, body_temps_init, air_temps_vec_init = sim.get_state()
air_temp_grid_init = np.array(air_temps_vec_init)

# Initial plot elements
im = ax.imshow(
    air_temp_grid_init.T,
    cmap="coolwarm",
    origin="lower",
    extent=[0, BOX_SIZE, 0, BOX_SIZE],
    interpolation="bilinear",
    animated=True,
)
scatter = ax.scatter(
    [p[0] for p in positions_init],
    [p[1] for p in positions_init],
    c=body_temps_init,
    cmap="viridis",
    edgecolor="k",
    # edgecolor="none",
    s=5,
    animated=True,
)
# 繪製初始企鵝速度方向 (quiver)
# quiver = ax.quiver(
#     [p[0] for p in positions_init],
#     [p[1] for p in positions_init],
#     [v[0] for v in velocities_init],
#     [v[1] for v in velocities_init],
#     scale=20,
#     scale_units="inches",
#     color="black",
#     headwidth=3,
#     headlength=4,
#     width=0.005,
#     animated=True,
# )

# Plot settings
ax.set_xlim(0, BOX_SIZE)
ax.set_ylim(0, BOX_SIZE)
ax.set_xlabel("X position")
ax.set_ylabel("Y position")
title = ax.set_title("Penguin Simulation - Step 0")
ax.set_aspect("equal", adjustable="box")
fig.colorbar(im, ax=ax, label="Air Temperature (°C)")
fig.colorbar(scatter, ax=ax, label="Penguin Body Temperature (°C)")


# --- Animation Update Function ---
def update(frame):
    global sim
    start_frame_time = time.time()

    # Run simulation steps for this frame
    for step_in_frame in range(STEPS_PER_FRAME):
        sim.step(DT)

    # Get current state
    positions, velocities, body_temps, air_temps_vec = sim.get_state()

    pos_array = np.array(positions)
    # vel_array = np.array(velocities)
    body_temps = np.array(body_temps)
    air_temp_grid = np.array(air_temps_vec).reshape((NUM_GRID, NUM_GRID))

    # --- Update Plot Data ---
    im.set_data(air_temp_grid.T)
    scatter.set_offsets(pos_array)
    scatter.set_array(body_temps)
    # quiver.set_offsets(pos_array)
    # quiver.set_UVC(vel_array[:, 0], vel_array[:, 1])

    # --- Update Color Limits Dynamically ---
    air_min, air_max = np.min(air_temp_grid), np.max(air_temp_grid)
    body_min, body_max = np.min(body_temps), np.max(body_temps)

    # --- Update Title and Print Status ---
    current_sim_time = (frame + 1) * STEPS_PER_FRAME * DT
    frame_time = time.time() - start_frame_time
    title_text = (
        f"Penguin Sim - Time: {current_sim_time:.2f}s Frame: {frame}/{TOTAL_FRAMES} "
        f"Body T: [{body_min:.2f}, {body_max:.2f}] Air T: [{air_min:.2f}, {air_max:.2f}] "
        f"(Frame time: {frame_time * 1000:.1f}ms)"
    )
    title.set_text(title_text)

    if air_min == air_max:
        air_min -= 0.5
        air_max += 0.5
    if body_min == body_max:
        body_min -= 0.5
        body_max += 0.5
    im_pad = (air_max - air_min) * 0.1
    sc_pad = (body_max - body_min) * 0.1
    im.set_clim(vmin=air_min - im_pad, vmax=air_max + im_pad)
    scatter.set_clim(vmin=body_min - sc_pad, vmax=body_max + sc_pad)

    print(f"\r{title_text}", end="")

    # return im, scatter, title, quiver
    return im, scatter, title  # , quiver


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
# ani.save("penguin_simulation.gif", writer="pillow", fps=FRAMES_PER_SECOND)
plt.show()

end_time = time.time()
print(f"\nSimulation and animation complete. Total time: {end_time - start_time:.2f}s")
print("GIF saved as penguin_simulation.gif")
