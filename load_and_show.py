import os
import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from util.animation import PenguinPlot

load_path = os.path.join("data", "N500_T100s_C(False)", "simulation.npz")
save_path = load_path.replace(".npz", ".mp4")
# save_path = None

# 讀取 npz 檔案
npz = np.load(load_path, allow_pickle=True)

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

FRAMES_PER_SECOND = 1.0 / (STEPS_PER_FRAME * DT)

NUM_FRAMES = 100
# NUM_FRAMES = positions.shape[0]
NUM_PENGUINS = positions.shape[1]

print(f"Loaded simulation data with {NUM_FRAMES} frames, {NUM_PENGUINS} penguins")
print(f"Grid size: {params['NUM_GRID']}x{params['NUM_GRID']}, Box size: {BOX_SIZE}")
print(f"Animation: Target FPS: {FRAMES_PER_SECOND:.2f}")


def make_title(current_t, frame, total_frames, air_temps, body_temps):
    air_stats = (np.min(air_temps), np.median(air_temps), np.max(air_temps))
    body_stats = (np.min(body_temps), np.median(body_temps), np.max(body_temps))
    return (
        f"Penguin Sim - Time: {current_t:.2f}s Frame: {frame}/{total_frames} "
        f"Body T: [{body_stats[0]:.2f}, {body_stats[1]:.2f},{body_stats[2]:.2f}] "
        f"Air T: [{air_stats[0]:.2f}, {air_stats[1]:.2f},{air_stats[2]:.2f}]"
    )


# --- Matplotlib Setup ---
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Initial state for plotting setup
positions_init = positions[0]
body_temps_init = body_temps[0]
air_temp_grid_init = air_temps[0]

# Initialize penguin plot
penguin_plot = PenguinPlot(
    ax,
    positions_init,
    body_temps_init,
    air_temp_grid_init,
    BOX_SIZE,
    PREFER_TEMP,
    NUM_FRAMES,
)

save_pbar = tqdm(total=NUM_FRAMES, desc="Rendering animation", unit="frame")


# --- Animation Update Function ---
def update(frame):
    # Get current state
    positions_frame = positions[frame]
    body_temps_frame = body_temps[frame]
    air_temp_grid = air_temps[frame]
    current_t = times[frame]

    # Update penguin plot
    penguin_artists = penguin_plot.update(
        positions_frame, body_temps_frame, air_temp_grid, current_t, frame
    )

    title_text = make_title(
        current_t, frame, NUM_FRAMES, air_temp_grid, body_temps_frame
    )

    print(f"\r{title_text}", end="\r")

    # 更新進度條
    save_pbar.update(1)

    # Return artists for blitting
    return penguin_artists


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
if save_path is not None:
    ani.save(save_path, writer="ffmpeg", fps=FRAMES_PER_SECOND)
else:
    plt.show()

save_pbar.close()
end_time = time.time()
print(f"\nAnimation rendering complete. Total time: {end_time - start_time:.2f}s")
