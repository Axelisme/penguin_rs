import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq

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

NUM_FRAMES = positions.shape[0]
NUM_PENGUINS = positions.shape[1]

print(f"Loaded simulation data with {NUM_FRAMES} frames, {NUM_PENGUINS} penguins")
print(f"Grid size: {NUM_GRID}x{NUM_GRID}, Box size: {BOX_SIZE}")

# Time domain analysis
print("\n=== Time Domain Analysis ===")

# Calculate sampling frequency and time step
dt = DT * STEPS_PER_FRAME  # Time step between frames
fs = 1.0 / dt  # Sampling frequency
print(f"Sampling frequency: {fs:.4f} Hz")
print(f"Time step between frames: {dt:.4f} s")
print(f"Total simulation time: {times[-1]:.2f} s")

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

# Calculate average amplitude spectrum across all penguins
avg_amplitude = np.mean(amplitudes, axis=0)

# Calculate frequency array
frequencies = fftfreq(NUM_FRAMES, dt)

# Only use positive frequencies (due to symmetry of real FFT)
positive_freq_mask = frequencies >= 0
frequencies_positive = frequencies[positive_freq_mask]
avg_amplitude_positive = avg_amplitude[positive_freq_mask]

# Create visualization
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: Body temperature time series for first few penguins
ax1.set_title("Body Temperature Time Series (First 5 Penguins)")
for i in range(min(5, NUM_PENGUINS)):
    ax1.plot(times, body_temps[:, i], label=f"Penguin {i + 1}", alpha=0.7)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Body Temperature (°C)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Average body temperature over time
avg_body_temp = np.mean(body_temps, axis=1)
ax2.set_title("Average Body Temperature Over Time")
ax2.plot(times, avg_body_temp, "b-", linewidth=2, label="Average Body Temperature")
ax2.axhline(
    y=PREFER_TEMP, color="r", linestyle="--", label=f"Preferred Temp ({PREFER_TEMP}°C)"
)
ax2.axhline(y=TEMP_ROOM, color="g", linestyle="--", label=f"Room Temp ({TEMP_ROOM}°C)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Body Temperature (°C)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Average amplitude spectrum
ax3.set_title("Average Amplitude Spectrum of Body Temperature Fluctuations")
ax3.plot(frequencies_positive, avg_amplitude_positive, "r-", linewidth=2)
ax3.set_xlabel("Frequency (Hz)")
ax3.set_ylabel("Average Amplitude")
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, min(10, fs / 2))  # Show up to 10 Hz or Nyquist frequency

plt.tight_layout()
plt.savefig("penguin_temperature_analysis.png", dpi=300, bbox_inches="tight")
plt.show()

# Print some statistics
print("\n=== Analysis Results ===")
print(f"Average body temperature: {np.mean(body_temps):.2f}°C")
print(f"Temperature standard deviation: {np.std(body_temps):.2f}°C")
print(f"Temperature range: [{np.min(body_temps):.2f}, {np.max(body_temps):.2f}]°C")

# Find dominant frequencies
dominant_freq_idx = np.argmax(avg_amplitude_positive[1:]) + 1  # Skip DC component
dominant_freq = frequencies_positive[dominant_freq_idx]
dominant_amplitude = avg_amplitude_positive[dominant_freq_idx]

print(f"\nDominant frequency: {dominant_freq:.4f} Hz")
print(f"Dominant frequency period: {1 / dominant_freq:.2f} s")
print(f"Dominant frequency amplitude: {dominant_amplitude:.4f}")

# Find top 5 frequencies with highest amplitudes (excluding DC)
freq_amplitude_pairs = list(zip(frequencies_positive[1:], avg_amplitude_positive[1:]))
freq_amplitude_pairs.sort(key=lambda x: x[1], reverse=True)

print("\nTop 5 dominant frequencies:")
for i, (freq, amp) in enumerate(freq_amplitude_pairs[:5]):
    print(
        f"{i + 1}. Frequency: {freq:.4f} Hz, Period: {1 / freq:.2f} s, Amplitude: {amp:.4f}"
    )
