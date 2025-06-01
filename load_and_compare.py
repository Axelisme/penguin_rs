import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from util import calculate_temp_gradient_relationship, get_env_temps_at_positions
from util.animation import (
    GradPlot,
    PenguinPlot,
    PhasePlot,
    VelocityRatioPlot,
)
from util.stable_temp import get_stable_point

# 讀取 npz 檔案
npz = np.load("penguin_simulation_data.npz", allow_pickle=True)

positions = npz["positions"]  # shape: (frames, N, 2)
body_temps = npz["body_temps"]  # shape: (frames, N)
air_temps = npz["air_temps"]  # shape: (frames, num_grid, num_grid)
times = npz["times"]  # shape: (frames,)
params = npz["params"].item()  # dict

# Extract parameters from saved data
DT = params["DT"]
BOX_SIZE = params["BOX_SIZE"]
STEPS_PER_FRAME = params["STEPS_PER_FRAME"]
PREFER_TEMP = params["PREFER_TEMP"]
TEMP_ROOM = params["TEMP_ROOM"]
NUM_GRID = params["NUM_GRID"]
HEAT_GEN_COEFF = params["HEAT_GEN_COEFF"]
HEAT_E2P_COEFF = params["HEAT_E2P_COEFF"]
PENGUIN_MOVE_FACTOR = params["PENGUIN_MOVE_FACTOR"]

# --- Temperature Unit Conversion ---
# Calculate stable penguin temperature to define new temperature unit
stable_body_temp, stable_env_temp = get_stable_point(
    PREFER_TEMP, HEAT_GEN_COEFF, HEAT_E2P_COEFF
)
# Use the difference between stable body temperature and room temperature as the new unit
temp_unit = stable_env_temp - TEMP_ROOM


def convert_temp_to_new_unit(temp_array):
    """Convert temperature from original unit to new unit (with TEMP_ROOM as origin)"""
    return (temp_array - TEMP_ROOM) / temp_unit


def convert_temp_diff_to_new_unit(temp_diff):
    """Convert temperature differences to new unit"""
    return temp_diff / temp_unit


print("Original temperature unit conversion:")
print(f"  TEMP_ROOM: {TEMP_ROOM:.2f} (original) -> 0.00 (new unit)")
print(
    f"  Stable body temp: {stable_body_temp:.2f} (original) -> {convert_temp_to_new_unit(stable_body_temp):.2f} (new unit)"
)
print(
    f"  Stable env temp: {stable_env_temp:.2f} (original) -> {convert_temp_to_new_unit(stable_env_temp):.2f} (new unit)"
)
print(f"  Temperature unit scale: 1 new unit = {temp_unit:.2f} original units")


# Convert all temperature data to new unit
body_temps_converted = convert_temp_to_new_unit(body_temps)
air_temps_converted = convert_temp_to_new_unit(air_temps)

# Convert temperature parameters to new unit
PREFER_TEMP_converted = convert_temp_to_new_unit(PREFER_TEMP)
TEMP_ROOM_converted = 0.0  # By definition, room temperature is 0 in new unit
HEAT_GEN_COEFF_converted = convert_temp_diff_to_new_unit(HEAT_GEN_COEFF)
HEAT_E2P_COEFF_converted = HEAT_E2P_COEFF
MOVE_FACTOR_converted = PENGUIN_MOVE_FACTOR * temp_unit**2


print("Converted parameters:")
print(f"  PREFER_TEMP: {PREFER_TEMP:.4f} -> {PREFER_TEMP_converted:.4f}")
print(f"  TEMP_ROOM: {TEMP_ROOM:.4f} -> {TEMP_ROOM_converted:.4f}")
print(f"  HEAT_GEN_COEFF: {HEAT_GEN_COEFF:.4f} -> {HEAT_GEN_COEFF_converted:.4f}")
print(f"  HEAT_E2P_COEFF: {HEAT_E2P_COEFF:.4f} -> {HEAT_E2P_COEFF_converted:.4f}")
print(
    f"  PENGUIN_MOVE_FACTOR: {PENGUIN_MOVE_FACTOR:.4f} -> {MOVE_FACTOR_converted:.4f}"
)

FRAMES_PER_SECOND = 1.0 / (STEPS_PER_FRAME * DT)

# NUM_FRAMES = positions.shape[0]
NUM_FRAMES = 100
NUM_PENGUINS = positions.shape[1]

print(f"Loaded simulation data with {NUM_FRAMES} frames, {NUM_PENGUINS} penguins")
print(f"Grid size: {NUM_GRID}x{NUM_GRID}, Box size: {BOX_SIZE}")
print(f"Animation: Target FPS: {FRAMES_PER_SECOND:.2f}")


def make_title(current_t, frame, total_frames, air_temps, body_temps):
    air_stats = (np.min(air_temps), np.median(air_temps), np.max(air_temps))
    body_stats = (np.min(body_temps), np.median(body_temps), np.max(body_temps))
    return (
        f"Penguin Sim - Time: {current_t:.2f}s Frame: {frame}/{total_frames} "
        f"Body T: [{body_stats[0]:.2f}, {body_stats[1]:.2f},{body_stats[2]:.2f}] (new unit) "
        f"Air T: [{air_stats[0]:.2f}, {air_stats[1]:.2f},{air_stats[2]:.2f}] (new unit)"
    )


# --- Matplotlib Setup ---
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
ax_scatter, ax_main = axs[0]
ax_velocity, ax_gradient = axs[1]

# Initial state for plotting setup (using converted temperature data)
positions_init = positions[0]
body_temps_init = body_temps_converted[0]
air_temp_grid_init = air_temps_converted[0]

# Calculate initial temperature-gradient relationship (using converted air temperature)
grad_temps_init, gradients_init = calculate_temp_gradient_relationship(
    air_temp_grid_init, BOX_SIZE
)

# Calculate environmental temperatures at penguin positions for PhasePlot (using converted air temperature)
env_temps_init = get_env_temps_at_positions(
    positions_init, air_temp_grid_init, BOX_SIZE
)

# Initialize plot objects with converted parameters
phase_plot = PhasePlot(
    ax_scatter,
    body_temps_init,
    env_temps_init,
    PREFER_TEMP_converted,
    HEAT_GEN_COEFF_converted,
    HEAT_E2P_COEFF_converted,
    TEMP_ROOM_converted,
)

penguin_plot = PenguinPlot(
    ax_main,
    positions_init,
    body_temps_init,
    air_temp_grid_init,
    BOX_SIZE,
    PREFER_TEMP_converted,
    NUM_FRAMES,
)

penguin_gradients_init = np.interp(env_temps_init, grad_temps_init, gradients_init)

vel_ratio_plot = VelocityRatioPlot(
    ax_velocity,
    body_temps_init,
    env_temps_init,
    penguin_gradients_init,
    MOVE_FACTOR_converted,
    PREFER_TEMP_converted,
    TEMP_ROOM_converted,
    DT * STEPS_PER_FRAME,
)

# Initialize gradient plot with converted parameters
grad_plot = GradPlot(
    ax_gradient,
    grad_temps_init,
    gradients_init,
    TEMP_ROOM_converted,
    PREFER_TEMP_converted,
)

save_pbar = tqdm(total=NUM_FRAMES, desc="Rendering animation", unit="frame")


# --- Animation Update Function ---
def update(frame):
    # Get current state (using converted temperature data)
    positions_frame = positions[frame]
    body_temps_frame = body_temps_converted[frame]
    air_temp_grid = air_temps_converted[frame]
    current_t = times[frame]

    # Calculate temperature gradient relationship (using converted air temperature)
    grad_temps, gradients = calculate_temp_gradient_relationship(
        air_temp_grid, BOX_SIZE
    )

    # Calculate environmental temperatures at penguin positions (using converted air temperature)
    env_temps = get_env_temps_at_positions(positions_frame, air_temp_grid, BOX_SIZE)
    penguin_gradients = np.interp(env_temps, grad_temps, gradients)

    # Update all plots
    phase_artists = phase_plot.update(body_temps_frame, env_temps)
    penguin_artists = penguin_plot.update(
        positions_frame, body_temps_frame, air_temp_grid, current_t, frame
    )
    vel_ratio_artists = vel_ratio_plot.update(
        body_temps_frame, env_temps, penguin_gradients
    )
    grad_artists = grad_plot.update(grad_temps, gradients)

    title_text = make_title(
        current_t, frame, NUM_FRAMES, air_temp_grid, body_temps_frame
    )

    print(f"\r{title_text}", end="\r")

    # 更新進度條
    save_pbar.update(1)

    # Return all artists for blitting
    return phase_artists + penguin_artists + vel_ratio_artists + grad_artists


# --- Save Animation as GIF ---
print("Rendering animation to video (this may take a while)...")
start_time = time.time()

ani = animation.FuncAnimation(
    fig,
    update,
    frames=NUM_FRAMES,
    interval=1000 / FRAMES_PER_SECOND,
    blit=True,
    repeat=False,
)

plt.tight_layout()
ani.save(
    "penguin_simulation_normal_unit.mp4",
    writer="ffmpeg",
    fps=FRAMES_PER_SECOND,
)

save_pbar.close()
end_time = time.time()
print(f"\nAnimation rendering complete. Total time: {end_time - start_time:.2f}s")
