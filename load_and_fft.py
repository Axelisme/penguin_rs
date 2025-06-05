import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq, ifft

load_path = os.path.join("data", "N500_T100s_C(True)", "simulation.npz")
save_path = load_path.replace(".npz", "_fft.png")
save_path = None


# 理論演化函數定義 (from evolution_system.py)
def density_func(x, PREFER_TEMP):
    """密度函數"""
    return 0.5 + np.arctan(0.7 * (x - PREFER_TEMP) * np.pi) / np.pi


def grad_func(y):
    """梯度函數"""
    return np.clip(-0.051 * (y + 6.2) ** 2 + 17, 0.0, None)


def calculate_theoretical_period(
    HEAT_GEN_COEFF, HEAT_E2P_COEFF, PENGUIN_MOVE_FACTOR, PREFER_TEMP
):
    """
    計算理論演化系統在穩定點附近的週期

    理論演化方程:
    dx/dt = HEAT_GEN_COEFF - HEAT_E2P_COEFF * (x - y)
    dy/dt = -PENGUIN_MOVE_FACTOR * density(x) * (x - PREFER_TEMP) * grad(y)**2

    Returns:
    --------
    theory_period : float
        理論週期 (秒)
    theory_freq : float
        理論頻率 (Hz)
    x_stable : float
        穩定點的 x 座標 (體溫)
    y_stable : float
        穩定點的 y 座標 (環境溫度)
    eigenvalues : complex array
        Jacobian 矩陣的特徵值
    """

    # 計算穩定點
    # 穩定點條件: dx/dt = 0 和 dy/dt = 0
    # dy/dt = 0 當 x = PREFER_TEMP (假設 density(x) != 0 且 grad(y) != 0)
    x_stable = PREFER_TEMP
    # dx/dt = 0: HEAT_GEN_COEFF - HEAT_E2P_COEFF * (x_stable - y_stable) = 0
    y_stable = x_stable - HEAT_GEN_COEFF / HEAT_E2P_COEFF

    # 計算穩定點處的函數值
    density_at_stable = density_func(x_stable, PREFER_TEMP)
    grad_at_stable = grad_func(y_stable)

    # 計算 Jacobian 矩陣
    # J = [∂(dx/dt)/∂x  ∂(dx/dt)/∂y]
    #     [∂(dy/dt)/∂x  ∂(dy/dt)/∂y]

    # ∂(dx/dt)/∂x = 0
    J11 = 0

    # ∂(dx/dt)/∂y = HEAT_E2P_COEFF
    J12 = HEAT_E2P_COEFF

    # ∂(dy/dt)/∂x = -PENGUIN_MOVE_FACTOR * [density'(x) * (x - PREFER_TEMP) + density(x)] * grad(y)**2
    # 在穩定點 x = PREFER_TEMP，所以 (x - PREFER_TEMP) = 0
    J21 = -PENGUIN_MOVE_FACTOR * density_at_stable * grad_at_stable**2

    # ∂(dy/dt)/∂y = -PENGUIN_MOVE_FACTOR * density(x) * (x - PREFER_TEMP) * 2 * grad(y) * grad'(y)
    # 在穩定點 x = PREFER_TEMP，所以 (x - PREFER_TEMP) = 0
    J22 = 0

    # Jacobian 矩陣
    J = np.array([[J11, J12], [J21, J22]])

    # 計算特徵值
    eigenvalues = np.linalg.eigvals(J)

    # 對於形如 [0 a; b 0] 的矩陣，特徵值是 ±√(ab)
    # 如果 ab < 0，特徵值是純虛數，系統有週期性振盪
    ab = J12 * J21

    if ab < 0:
        # 週期性振盪
        imaginary_part = np.sqrt(-ab)
        theory_freq = imaginary_part / (2 * np.pi)  # Hz
        theory_period = 2 * np.pi / imaginary_part  # seconds
        is_oscillatory = True
    else:
        # 不穩定或非振盪
        theory_freq = None
        theory_period = None
        is_oscillatory = False

    return {
        "period": theory_period,
        "frequency": theory_freq,
        "x_stable": x_stable,
        "y_stable": y_stable,
        "eigenvalues": eigenvalues,
        "jacobian": J,
        "is_oscillatory": is_oscillatory,
        "ab_product": ab,
    }


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
TEMP_ROOM = params["TEMP_ROOM"]
NUM_GRID = params["NUM_GRID"]
HEAT_GEN_COEFF = params["HEAT_GEN_COEFF"]
HEAT_E2P_COEFF = params["HEAT_E2P_COEFF"]
PENGUIN_MOVE_FACTOR = params["PENGUIN_MOVE_FACTOR"]

NUM_FRAMES = positions.shape[0]
NUM_PENGUINS = positions.shape[1]

print(f"Loaded simulation data with {NUM_FRAMES} frames, {NUM_PENGUINS} penguins")
print(f"Grid size: {NUM_GRID}x{NUM_GRID}, Box size: {BOX_SIZE}")

# Calculate theoretical period
print("\n=== Theoretical Analysis ===")
theory_results = calculate_theoretical_period(
    HEAT_GEN_COEFF, HEAT_E2P_COEFF, PENGUIN_MOVE_FACTOR, PREFER_TEMP
)

print(
    f"Stable point: ({theory_results['x_stable']:.2f}, {theory_results['y_stable']:.2f})"
)
print(f"Jacobian eigenvalues: {theory_results['eigenvalues']}")
print(f"ab product: {theory_results['ab_product']:.6f}")

if theory_results["is_oscillatory"]:
    print(f"Theoretical frequency: {theory_results['frequency']:.6f} Hz")
    print(f"Theoretical period: {theory_results['period']:.2f} s")
else:
    print("System is not oscillatory (no periodic solution)")

# Time domain analysis
print("\n=== Time Domain Analysis ===")

# Calculate sampling frequency and time step
dt = DT * STEPS_PER_FRAME  # Time step between frames
fs = 1.0 / dt  # Sampling frequency
print(f"Sampling frequency: {fs:.4f} Hz")
print(f"Time step between frames: {dt:.4f} s")
print(f"Total simulation time: {times[-1]:.2f} s")

# Define frequency range of interest (0.1s to 1000s periods)
freq_min = 1.0 / 1000.0  # 0.001 Hz (1000s period)
freq_max = 1.0 / 10.0  # 0.1 Hz (10s period)
print(f"Analyzing frequency range: {freq_min:.4f} - {freq_max:.4f} Hz")
print(f"Corresponding period range: {1 / freq_max:.1f} - {1 / freq_min:.1f} s")

# Perform FFT analysis for each penguin's body temperature
print("\nPerforming FFT analysis...")

# Initialize arrays to store FFT results
fft_results = np.zeros((NUM_PENGUINS, NUM_FRAMES), dtype=complex)
amplitudes = np.zeros((NUM_PENGUINS, NUM_FRAMES))

# Calculate FFT for each penguin
for i in range(NUM_PENGUINS):
    # Get body temperature time series for penguin i
    temp_series = body_temps[:, i]

    # Remove DC component (mean temperature)
    temp_series_centered = temp_series - np.mean(temp_series)

    # Apply FFT
    fft_result = fft(temp_series_centered)
    fft_results[i, :] = fft_result

    # Calculate amplitude spectrum
    amplitudes[i, :] = np.abs(fft_result)

# Calculate frequency array
frequencies = fftfreq(NUM_FRAMES, dt)

# Focus on positive frequencies within our range of interest
positive_freq_mask = (frequencies >= freq_min) & (frequencies <= freq_max)
frequencies_of_interest = frequencies[positive_freq_mask]
amplitudes_of_interest = amplitudes[:, positive_freq_mask]

# Calculate average amplitude spectrum in the frequency range of interest
avg_amplitude_of_interest = np.mean(amplitudes_of_interest, axis=0)

# Find the dominant frequency in our range of interest
dominant_freq_idx_local = np.argmax(avg_amplitude_of_interest)
dominant_freq = frequencies_of_interest[dominant_freq_idx_local]
dominant_freq_idx_global = np.where(frequencies == dominant_freq)[0][0]

print(f"\nDominant frequency in range: {dominant_freq:.6f} Hz")
print(f"Dominant frequency period: {1 / dominant_freq:.2f} s")
print(
    f"Dominant frequency amplitude: {avg_amplitude_of_interest[dominant_freq_idx_local]:.4f}"
)

# Phase alignment procedure (using sum of all frequencies)
print("\nPerforming phase alignment using sum of all frequencies...")

# Calculate the sum of all frequencies for each penguin
fft_sum_per_penguin = np.sum(fft_results, axis=1)

# Calculate phases based on the sum of all frequencies
phases = np.angle(fft_sum_per_penguin)
amplitudes_at_sum = np.abs(fft_sum_per_penguin)

# Find the reference phase (use the penguin with the largest amplitude sum)
ref_penguin_idx = np.argmax(amplitudes_at_sum)
ref_phase = phases[ref_penguin_idx]

print(f"Reference penguin: {ref_penguin_idx} with phase {ref_phase:.4f} rad")

# Align all penguins' phases to the reference phase
phase_shifts = ref_phase - phases
aligned_fft_results = np.zeros_like(fft_results)

for i in range(NUM_PENGUINS):
    # Apply phase shift to the entire spectrum
    aligned_fft_results[i, :] = fft_results[i, :] * np.exp(1j * phase_shifts[i])

# Calculate the aligned average FFT
avg_aligned_fft = np.mean(aligned_fft_results, axis=0)
avg_aligned_amplitude = np.abs(avg_aligned_fft)

# Convert aligned average FFT back to time domain
avg_aligned_time_series = ifft(avg_aligned_fft).real

# Create visualization with 4 plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Body temperature time series for first few penguins
ax1.set_title("Body Temperature Time Series (First 5 Penguins)")
for i in range(min(5, NUM_PENGUINS)):
    ax1.plot(times, body_temps[:, i], label=f"Penguin {i + 1}", alpha=0.7)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Body Temperature (°C)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Amplitude spectrum in frequency range of interest (before alignment)
ax2.set_title(
    f"Average Amplitude Spectrum Before Alignment ({1 / freq_max:.0f}s - {1 / freq_min:.0f}s periods)"
)
ax2.plot(
    frequencies_of_interest,
    avg_amplitude_of_interest,
    "r-",
    linewidth=2,
    label="Before alignment",
)
ax2.axvline(
    x=dominant_freq,
    color="k",
    linestyle="--",
    label=f"Dominant freq: {dominant_freq:.4f} Hz",
)
# Add theoretical frequency line if oscillatory
if theory_results["is_oscillatory"] and theory_results["frequency"] is not None:
    if freq_min <= theory_results["frequency"] <= freq_max:
        ax2.axvline(
            x=theory_results["frequency"],
            color="purple",
            linestyle="-.",
            linewidth=2,
            label=f"Theory freq: {theory_results['frequency']:.4f} Hz",
        )
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Average Amplitude")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Amplitude spectrum after phase alignment
aligned_amplitudes_of_interest = avg_aligned_amplitude[positive_freq_mask]
ax3.set_title("Average Amplitude Spectrum After Phase Alignment")
ax3.plot(
    frequencies_of_interest,
    aligned_amplitudes_of_interest,
    "g-",
    linewidth=2,
    label="After alignment",
)
ax3.axvline(
    x=dominant_freq,
    color="k",
    linestyle="--",
    label=f"Dominant freq: {dominant_freq:.4f} Hz",
)
# Add theoretical frequency line if oscillatory
if theory_results["is_oscillatory"] and theory_results["frequency"] is not None:
    if freq_min <= theory_results["frequency"] <= freq_max:
        ax3.axvline(
            x=theory_results["frequency"],
            color="purple",
            linestyle="-.",
            linewidth=2,
            label=f"Theory freq: {theory_results['frequency']:.4f} Hz",
        )
ax3.set_xlabel("Frequency (Hz)")
ax3.set_ylabel("Aligned Average Amplitude")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Reconstructed time series from aligned average FFT
avg_body_temp = np.mean(body_temps, axis=1)
ax4.set_title("Reconstructed Signal from Phase-Aligned Average FFT")
ax4.plot(
    times,
    avg_aligned_time_series,
    "purple",
    linewidth=2,
    label="Aligned average signal",
)
ax4.plot(
    times,
    avg_body_temp - np.mean(avg_body_temp),
    "b--",
    alpha=0.7,
    label="Original average (centered)",
)
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Temperature Fluctuation (°C)")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
if save_path is not None:
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
else:
    plt.show()
