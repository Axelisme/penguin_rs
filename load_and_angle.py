import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

load_path = os.path.join("data", "N500_T100s_C(True)", "simulation.npz")
save_path = load_path.replace(".npz", "_angle.mp4")
save_path = None


def calculate_center_of_mass(positions):
    """Calculate the center of mass of all penguins"""
    return np.mean(positions, axis=0)


def calculate_angles_from_center(positions, center):
    """Calculate angles of each penguin relative to center of mass"""
    # Vector from center to each penguin
    relative_positions = positions - center

    # Calculate angles using atan2 (returns values from -π to π)
    angles = np.arctan2(relative_positions[:, 1], relative_positions[:, 0])

    # Convert to degrees and normalize to [0, 360)
    angles_degrees = np.degrees(angles)
    angles_degrees = (angles_degrees + 360) % 360

    return angles_degrees


def calculate_temp_difference(body_temps, prefer_temp):
    """Calculate body temperature difference from preferred temperature"""
    return body_temps - prefer_temp


def main():
    # Load npz file
    npz = np.load(load_path, allow_pickle=True)

    positions = npz["positions"]  # shape: (frames, N, 2)
    body_temps = npz["body_temps"]  # shape: (frames, N)
    times = npz["times"]  # shape: (frames,)
    params = npz["params"].item()  # dict

    PREFER_TEMP = params["PREFER_TEMP"]

    print(
        f"Loaded simulation data with {positions.shape[0]} frames, {positions.shape[1]} penguins"
    )
    print(f"Preferred temperature: {PREFER_TEMP}°C")

    # Filter out first 50 seconds
    time_threshold = 50.0
    start_frame_idx = np.searchsorted(times, time_threshold)

    # Use data after 50 seconds
    positions_filtered = positions[start_frame_idx:]
    body_temps_filtered = body_temps[start_frame_idx:]
    times_filtered = times[start_frame_idx:]

    print(f"Using {len(positions_filtered)} frames after {time_threshold}s")

    # Downsample for animation performance
    downsample_factor = max(1, len(positions_filtered) // 200)  # Target ~200 frames
    frame_indices = np.arange(0, len(positions_filtered), downsample_factor)

    positions_anim = positions_filtered[frame_indices]
    body_temps_anim = body_temps_filtered[frame_indices]
    times_anim = times_filtered[frame_indices]

    print(f"Animation will have {len(positions_anim)} frames")

    # Prepare data for animation
    animation_data = []

    for frame_idx in range(len(positions_anim)):
        curr_positions = positions_anim[frame_idx]
        curr_body_temps = body_temps_anim[frame_idx]
        curr_time = times_anim[frame_idx]

        # Calculate center of mass
        center_of_mass = calculate_center_of_mass(curr_positions)

        # Calculate angles relative to center of mass
        angles = calculate_angles_from_center(curr_positions, center_of_mass)

        # Calculate temperature difference from preferred temperature
        temp_diff = calculate_temp_difference(curr_body_temps, PREFER_TEMP)

        animation_data.append(
            {
                "time": curr_time,
                "angles": angles,
                "temp_diff": temp_diff,
                "center_of_mass": center_of_mass,
            }
        )

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Initialize empty scatter plot
    scatter = ax.scatter([], [], alpha=0.7, s=20, c="blue")

    # Set up plot properties
    ax.set_xlabel("Angle relative to Center of Mass (degrees)")
    ax.set_ylabel("Body Temperature - Preferred Temperature (°C)")
    ax.set_xlim(0, 360)

    # Calculate y-axis limits based on all data
    all_temp_diffs = np.concatenate([data["temp_diff"] for data in animation_data])
    y_margin = 0.1 * (np.max(all_temp_diffs) - np.min(all_temp_diffs))
    ax.set_ylim(np.min(all_temp_diffs) - y_margin, np.max(all_temp_diffs) + y_margin)

    ax.grid(True, alpha=0.3)
    ax.axhline(
        y=0, color="red", linestyle="--", alpha=0.5, label="Preferred Temperature"
    )
    ax.legend()

    # Title that will be updated
    title = ax.set_title("")

    def animate(frame):
        """Animation function called for each frame"""
        data = animation_data[frame]

        # Update scatter plot data
        scatter.set_offsets(np.column_stack([data["angles"], data["temp_diff"]]))

        # Color points based on temperature difference
        colors = plt.cm.RdBu_r(
            (data["temp_diff"] - np.min(all_temp_diffs))
            / (np.max(all_temp_diffs) - np.min(all_temp_diffs))
        )
        scatter.set_color(colors)

        # Update title with current time
        title.set_text(
            f"Penguin Distribution: Angle vs Temperature Difference\nTime: {data['time']:.1f}s"
        )

        return scatter, title

    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=len(animation_data), interval=100, blit=False, repeat=True
    )

    # Save animation if save_path is specified
    if save_path:
        print(f"Saving animation to {save_path}...")
        writer = animation.FFMpegWriter(
            fps=10, metadata=dict(artist="Penguin Simulation"), bitrate=1800
        )
        anim.save(save_path, writer=writer)
        print(f"Animation saved to {save_path}")

    # Show animation
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
