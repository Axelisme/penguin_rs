import os
import time

import numpy as np
from penguin_rs import PySimulation
from tqdm.auto import tqdm

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
SIM_TIME = 100.0
DT = 0.003
TOTAL_STEPS = int(SIM_TIME / DT)
STEPS_PER_FRAME = 100
TOTAL_FRAMES = int(TOTAL_STEPS / STEPS_PER_FRAME)

# 數據儲存設定
data_times = []
data_positions = []
data_velocities = []
data_body_temps = []
data_air_temps = []

# --- 開始模擬並儲存數據 ---
print("開始模擬並儲存數據 (這可能需要一段時間)...")
start_time = time.time()

with tqdm(total=TOTAL_STEPS) as t:
    for step in range(TOTAL_STEPS):
        sim.step(DT)

        # 每隔SAVE_FRAME_INTERVAL步保存一次數據
        if step % STEPS_PER_FRAME == 0:
            frame = step // STEPS_PER_FRAME

            start_frame_time = time.time()
            positions, velocities, body_temps, air_temps = sim.get_state()

            # 儲存數據
            current_sim_time = step * DT
            data_times.append(current_sim_time)
            data_positions.append(positions)
            data_velocities.append(velocities)
            data_body_temps.append(body_temps)
            data_air_temps.append(air_temps)

            # 計算溫度範圍
            air_min, air_max = np.min(air_temps), np.max(air_temps)
            body_min, body_max = np.min(body_temps), np.max(body_temps)

            # 計算並輸出標題文字
            frame_time = time.time() - start_frame_time
            title_text = (
                f"Time: {current_sim_time:.2f}s Frame: {frame}/{TOTAL_FRAMES} "
                f"Body T: [{body_min:.2f}, {body_max:.2f}] Air T: [{air_min:.2f}, {air_max:.2f}] "
                f"(Frame time: {frame_time * 1000:.1f}ms)"
            )
            t.set_description(title_text)
        t.update(1)

end_time = time.time()
print(f"\n\n模擬完成。總時間: {end_time - start_time:.2f}秒")

# 將數據轉換為 NumPy 陣列
data_times = np.array(data_times)
data_positions = np.array(data_positions)
data_velocities = np.array(data_velocities)
data_body_temps = np.array(data_body_temps)
data_air_temps = np.array(data_air_temps)

print("開始儲存數據到 .npz 檔案...")

# 儲存數據到 .npz 檔案
filename = "penguin_simulation_data"
# Determining output filename; append index if file already exists
index = 0
output_filename = f"{filename}.npz"
while os.path.exists(output_filename):
    index += 1
    output_filename = f"{filename}_{index}.npz"
np.savez_compressed(
    output_filename,
    times=data_times,
    positions=data_positions,
    velocities=data_velocities,
    body_temps=data_body_temps,
    air_temps=data_air_temps,
    params={
        "SEED": SEED,
        "NUM_PENGUINS": NUM_PENGUINS,
        "PENGUIN_MOVE_FACTOR": PENGUIN_MOVE_FACTOR,
        "PENGUIN_RADIUS": PENGUIN_RADIUS,
        "HEAT_GEN_COEFF": HEAT_GEN_COEFF,
        "HEAT_P2E_COEFF": HEAT_P2E_COEFF,
        "HEAT_E2P_COEFF": HEAT_E2P_COEFF,
        "INIT_TEMP_MEAN": INIT_TEMP_MEAN,
        "PREFER_TEMP": PREFER_TEMP,
        "NUM_GRID": NUM_GRID,
        "BOX_SIZE": BOX_SIZE,
        "DIFFUSION_COEFF": DIFFUSION_COEFF,
        "DECAY_COEFF": DECAY_COEFF,
        "TEMP_ROOM": TEMP_ROOM,
        "COLLISION_STRENGTH": COLLISION_STRENGTH,
        "DENSITY_FACTOR": DENSITY_FACTOR,
        "SIM_TIME": SIM_TIME,
        "DT": DT,
        "TOTAL_STEPS": TOTAL_STEPS,
        "STEPS_PER_FRAME": STEPS_PER_FRAME,
        "TOTAL_FRAMES": TOTAL_FRAMES,
    },
)

print(f"模擬數據已儲存至 {output_filename}")
print(f"儲存了 {len(data_times)} 個時間點的數據")
print("每個時間點包含: 企鵝位置, 企鵝速度, 企鵝體溫, 環境溫度網格")
