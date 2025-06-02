import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq, ifft

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

# Define frequency range of interest (1s to 1000s periods)
freq_min = 1.0 / 1000.0  # 0.001 Hz (1000s period)
freq_max = 1.0 / 1.0  # 1 Hz (1s period)
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

# Phase alignment procedure
print("\nPerforming phase alignment...")

# Extract the complex values at the dominant frequency for all penguins
dominant_freq_complex = fft_results[:, dominant_freq_idx_global]

# Calculate phases at the dominant frequency
phases = np.angle(dominant_freq_complex)
amplitudes_at_dominant = np.abs(dominant_freq_complex)

# Find the reference phase (use the penguin with the largest amplitude)
ref_penguin_idx = np.argmax(amplitudes_at_dominant)
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

# Create visualization
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 12))

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

# Plot 3: Amplitude spectrum in frequency range of interest (before alignment)
ax3.set_title(
    f"Average Amplitude Spectrum ({1 / freq_max:.0f}s - {1 / freq_min:.0f}s periods)"
)
ax3.plot(
    frequencies_of_interest,
    avg_amplitude_of_interest,
    "r-",
    linewidth=2,
    label="Before alignment",
)
ax3.axvline(
    x=dominant_freq,
    color="k",
    linestyle="--",
    label=f"Dominant freq: {dominant_freq:.4f} Hz",
)
ax3.set_xlabel("Frequency (Hz)")
ax3.set_ylabel("Average Amplitude")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Amplitude spectrum after phase alignment
aligned_amplitudes_of_interest = avg_aligned_amplitude[positive_freq_mask]
ax4.set_title("Amplitude Spectrum After Phase Alignment")
ax4.plot(
    frequencies_of_interest,
    aligned_amplitudes_of_interest,
    "g-",
    linewidth=2,
    label="After alignment",
)
ax4.axvline(
    x=dominant_freq,
    color="k",
    linestyle="--",
    label=f"Dominant freq: {dominant_freq:.4f} Hz",
)
ax4.set_xlabel("Frequency (Hz)")
ax4.set_ylabel("Aligned Average Amplitude")
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Phase distribution before and after alignment
aligned_phases = np.angle(aligned_fft_results[:, dominant_freq_idx_global])
ax5.set_title("Phase Distribution at Dominant Frequency")
ax5.hist(phases, bins=20, alpha=0.7, label="Before alignment", density=True)

# Check if aligned phases have sufficient range for histogram
phase_range = np.max(aligned_phases) - np.min(aligned_phases)
if phase_range > 1e-6:  # If there's enough variation
    ax5.hist(
        aligned_phases,
        bins=min(20, max(5, int(NUM_PENGUINS / 25))),
        alpha=0.7,
        label="After alignment",
        density=True,
    )
else:  # If phases are too close, just show a vertical line
    ax5.axvline(
        x=np.mean(aligned_phases),
        color="orange",
        linewidth=3,
        alpha=0.7,
        label="After alignment (all aligned)",
    )

ax5.axvline(
    x=ref_phase, color="r", linestyle="--", label=f"Reference phase: {ref_phase:.3f}"
)
ax5.set_xlabel("Phase (radians)")
ax5.set_ylabel("Density")
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Reconstructed time series from aligned average FFT
ax6.set_title("Reconstructed Signal from Phase-Aligned Average FFT")
ax6.plot(
    times,
    avg_aligned_time_series,
    "purple",
    linewidth=2,
    label="Aligned average signal",
)
ax6.plot(
    times,
    avg_body_temp - np.mean(avg_body_temp),
    "b--",
    alpha=0.7,
    label="Original average (centered)",
)
ax6.set_xlabel("Time (s)")
ax6.set_ylabel("Temperature Fluctuation (°C)")
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("penguin_phase_aligned_analysis.png", dpi=300, bbox_inches="tight")
plt.show()

# Print detailed analysis results
print("\n=== Analysis Results ===")
print(f"Average body temperature: {np.mean(body_temps):.2f}°C")
print(f"Temperature standard deviation: {np.std(body_temps):.2f}°C")
print(f"Temperature range: [{np.min(body_temps):.2f}, {np.max(body_temps):.2f}]°C")

print("\n=== Phase Alignment Results ===")
print(f"Dominant frequency: {dominant_freq:.6f} Hz")
print(f"Dominant frequency period: {1 / dominant_freq:.2f} s")
print(
    f"Amplitude before alignment: {avg_amplitude_of_interest[dominant_freq_idx_local]:.4f}"
)
print(
    f"Amplitude after alignment: {aligned_amplitudes_of_interest[dominant_freq_idx_local]:.4f}"
)
print(
    f"Amplitude enhancement ratio: {aligned_amplitudes_of_interest[dominant_freq_idx_local] / avg_amplitude_of_interest[dominant_freq_idx_local]:.2f}"
)

# Phase coherence analysis
phase_std_before = np.std(phases)
phase_std_after = np.std(aligned_phases)
print(f"Phase standard deviation before alignment: {phase_std_before:.4f} rad")
print(f"Phase standard deviation after alignment: {phase_std_after:.4f} rad")

# Find top 5 frequencies in our range of interest
freq_amplitude_pairs = list(zip(frequencies_of_interest, avg_amplitude_of_interest))
freq_amplitude_pairs.sort(key=lambda x: x[1], reverse=True)

print("\nTop 5 dominant frequencies in range:")
for i, (freq, amp) in enumerate(freq_amplitude_pairs[:5]):
    print(
        f"{i + 1}. Frequency: {freq:.6f} Hz, Period: {1 / freq:.2f} s, Amplitude: {amp:.4f}"
    )
