import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

# 讀取 npz 檔案
npz = np.load("penguin_simulation_data.npz", allow_pickle=True)

positions = npz["positions"]  # shape: (frames, N, 2)
body_temps = npz["body_temps"]  # shape: (frames, N)
air_temps = npz["air_temps"]  # shape: (frames, num_grid, num_grid)
times = npz["times"]  # shape: (frames,)
params = npz["params"].item()  # dict


DT = params["DT"]
BOX_SIZE = params["BOX_SIZE"]
STEPS_PER_FRAME = params["STEPS_PER_FRAME"]
FRAME_PER_SECOND = 1.0 / (STEPS_PER_FRAME * DT)


NUM_FRAMES = positions.shape[0]
NUM_GRID = air_temps.shape[1]

fig, ax = plt.subplots(figsize=(8, 8))

im = ax.imshow(
    air_temps[0].T,
    cmap="coolwarm",
    origin="lower",
    extent=[0, BOX_SIZE, 0, BOX_SIZE],
    interpolation="bilinear",
    animated=True,
)
sc = ax.scatter(
    positions[0, :, 0],
    positions[0, :, 1],
    c=body_temps[0],
    cmap="viridis",
    edgecolor="k",
    s=5,
    animated=True,
)
ax.set_xlim(0, BOX_SIZE)
ax.set_ylim(0, BOX_SIZE)
ax.set_xlabel("X position")
ax.set_ylabel("Y position")
title = ax.set_title(f"Penguin Simulation - Time: {times[0]:.2f}s")
ax.set_aspect("equal", adjustable="box")
fig.colorbar(im, ax=ax, label="Air Temperature (°C)")
fig.colorbar(sc, ax=ax, label="Penguin Body Temperature (°C)")


save_pbar = tqdm(total=NUM_FRAMES, desc="Saving GIF", unit="frame")


# 動畫更新函數
def update(frame):
    im.set_data(air_temps[frame].T)
    sc.set_offsets(positions[frame])
    sc.set_array(body_temps[frame])
    title.set_text(f"Penguin Simulation - Time: {times[frame]:.2f}s")

    air_min, air_max = np.min(air_temps[frame]), np.max(air_temps[frame])
    body_min, body_max = np.min(body_temps[frame]), np.max(body_temps[frame])

    if air_min == air_max:
        air_min -= 0.5
        air_max += 0.5
    if body_min == body_max:
        body_min -= 0.5
        body_max += 0.5
    im_pad = (air_max - air_min) * 0.1
    sc_pad = (body_max - body_min) * 0.1
    im.set_clim(vmin=air_min - im_pad, vmax=air_max + im_pad)
    sc.set_clim(vmin=body_min - sc_pad, vmax=body_max + sc_pad)

    # 更新進度條
    save_pbar.update(1)

    return im, sc, title


ani = animation.FuncAnimation(
    fig,
    update,
    frames=NUM_FRAMES,
    interval=1,
    # blit=True,
    repeat=False,
)

plt.tight_layout()
ani.save("penguin_simulation.mp4", writer="ffmpeg", fps=FRAME_PER_SECOND)

save_pbar.close()
print("MP4 儲存完成！")
