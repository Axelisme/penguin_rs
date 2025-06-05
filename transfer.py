import os

import numpy as np

load_path = os.path.join("data", "N500_T100s_C(True)", "simulation.npz")

# 讀取 npz 檔案
npz = np.load(load_path, allow_pickle=True)

params = npz["params"].item()

params["PREFER_TEMP"] = params["PREFER_TEMP_COMMON"]
del params["PREFER_TEMP_COMMON"]

np.savez_compressed(
    load_path,
    times=npz["times"],
    positions=npz["positions"],
    velocities=npz["velocities"],
    body_temps=npz["body_temps"],
    air_temps=npz["air_temps"],
    params=params,
)
