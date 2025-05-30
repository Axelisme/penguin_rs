import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

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

# 讀取 npz 檔案
npz = np.load("penguin_simulation_collision_False.npz", allow_pickle=True)

positions = npz["positions"]  # shape: (frames, N, 2)
body_temps = npz["body_temps"]  # shape: (frames, N)
air_temps = npz["air_temps"]  # shape: (frames, num_grid, num_grid)
times = npz["times"]  # shape: (frames,)
params = npz["params"].item()  # dict

# Extract parameters
DT = params["DT"]
BOX_SIZE = params["BOX_SIZE"]
STEPS_PER_FRAME = params["STEPS_PER_FRAME"]
PREFER_TEMP_COMMON = params["PREFER_TEMP_COMMON"]
TEMP_ROOM = params["TEMP_ROOM"]
NUM_GRID = params["NUM_GRID"]

FRAME_PER_SECOND = 1.0 / (STEPS_PER_FRAME * DT)

NUM_FRAMES = positions.shape[0]
NUM_PENGUINS = positions.shape[1]

print(f"Loaded simulation data with {NUM_FRAMES} frames, {NUM_PENGUINS} penguins")
print(f"Grid size: {NUM_GRID}x{NUM_GRID}, Box size: {BOX_SIZE}")
print(f"Animation: Target FPS: {FRAME_PER_SECOND:.2f}")


# --- Matplotlib Setup ---
fig = plt.figure(figsize=(16, 12))
ax_scatter = plt.subplot(2, 2, 1)  # Top left
ax_main = plt.subplot(2, 2, 2)  # Top right
ax_density = plt.subplot(2, 2, 3)  # Bottom left
ax_gradient = plt.subplot(2, 2, 4)  # Bottom right

# Initial state for plotting setup
positions_init = positions[0]
body_temps_init = body_temps[0]
air_temp_grid_init = air_temps[0]

# Calculate initial temperature-gradient relationship
temp_range_init, avg_gradients_init = calculate_temp_gradient_relationship(
    air_temp_grid_init
)

# Calculate initial penguin density by temperature
temp_range_density_init, densities_init = calculate_penguin_density_by_temp(
    positions_init, air_temp_grid_init, BOX_SIZE, NUM_GRID
)

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
title = ax_main.set_title("Penguin Simulation - Frame 0")
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

save_pbar = tqdm(total=NUM_FRAMES, desc="Saving animation", unit="frame")


# --- Animation Update Function ---
def update(frame):
    # Get current state
    positions_frame = positions[frame]
    body_temps_frame = body_temps[frame]
    air_temp_grid = air_temps[frame]

    # Create binary colors based on temperature preference
    penguin_colors = create_penguin_colors(body_temps_frame, PREFER_TEMP_COMMON)

    # Calculate environmental temperatures at current penguin positions
    env_temps = get_env_temps_at_positions(
        positions_frame, air_temp_grid, BOX_SIZE, NUM_GRID
    )

    # --- Update Plot Data ---
    # Update main simulation view
    im.set_data(air_temp_grid.T)

    scatter_main.set_offsets(positions_frame)
    scatter_main.set_color(penguin_colors)

    # Update temperature scatter plot
    scatter_temp.set_offsets(np.column_stack([body_temps_frame, env_temps]))

    # Update gradient plot
    temp_range, avg_gradients = calculate_temp_gradient_relationship(air_temp_grid)
    gradient_line.set_data(temp_range, avg_gradients)

    # Update density plot
    temp_range_density, densities = calculate_penguin_density_by_temp(
        positions_frame, air_temp_grid, BOX_SIZE, NUM_GRID
    )
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
    air_stats, body_stats = calculate_temperature_stats(air_temp_grid, body_temps_frame)
    air_min, air_mid, air_max = air_stats
    body_min, body_mid, body_max = body_stats

    ax_scatter.set_xlim(body_min - 1, body_max + 1)
    ax_scatter.set_ylim(air_min - 1, air_max + 1)

    # --- Update Title ---
    current_sim_time = times[frame]
    title_text = create_title_text(
        current_sim_time, frame, NUM_FRAMES, air_stats, body_stats
    )
    title.set_text(title_text)

    update_color_limits(im, air_temp_grid)

    # 更新進度條
    save_pbar.update(1)

    return im, scatter_main, scatter_temp, gradient_line, density_line, title


# Create animation
ani = animation.FuncAnimation(
    fig,
    update,
    frames=NUM_FRAMES,
    interval=1000 / FRAME_PER_SECOND,
    blit=True,
    repeat=False,
)

plt.tight_layout()
ani.save("penguin_simulation_loaded.mp4", writer="ffmpeg", fps=FRAME_PER_SECOND)

save_pbar.close()
print("動畫儲存完成！")
