import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation, PillowWriter
from pca_svd import PCA

class BranchPropertyEstimator:
    """
    Estimates natural frequency and damping ratio of tree branches
    from 3D marker motion trajectories using FFT and half-power bandwidth method.
    """
    
    def __init__(self, sampling_rate):
        """
        Args:
            sampling_rate: Sampling frequency of the motion data (Hz)
        """
        self.sampling_rate = sampling_rate
        
    def calculate_deflection_angle(self, parent_markers, child_markers):
        """
        Calculate relative deflection angle between sub-branch and parent branch.
        
        Args:
            parent_markers: Array of shape (n_frames, 3) - parent branch marker positions
            child_markers: Array of shape (n_frames, 3) - child branch marker positions
            
        Returns:
            theta: Array of deflection angles over time (radians)
        """
        # Calculate vectors from parent to child at each time step
        vectors = child_markers - parent_markers
        
        # Calculate deflection angle relative to initial position
        initial_vector = vectors[0]
        initial_vector_norm = initial_vector / np.linalg.norm(initial_vector)
        
        theta = np.zeros(len(vectors))
        for i, vec in enumerate(vectors):
            vec_norm = vec / np.linalg.norm(vec)
            # Calculate angle between current and initial vector
            dot_product = np.clip(np.dot(vec_norm, initial_vector_norm), -1.0, 1.0)
            theta[i] = np.arccos(dot_product)
            
        return theta
    
    def pca_transform(self, parent_markers, child_markers):
        vectors = child_markers - parent_markers
        pca = PCA(n_components=1)
        pca_vector = pca.fit_transform(vectors)
        return pca_vector[:, 0]

    def estimate_natural_frequency(self, theta_t, visualize=False):
        """
        Estimate natural frequency using FFT.
        
        Args:
            theta_t: Deflection angle signal over time
            visualize: Whether to plot the frequency spectrum
            
        Returns:
            f0: Natural frequency (Hz)
            fft_freq: Frequency array
            fft_power: Power spectrum
        """
        n = len(theta_t)
        
        # Perform FFT
        fft_result = fft(theta_t)
        fft_freq = fftfreq(n, 1/self.sampling_rate)
        
        # Take only positive frequencies
        positive_freq_mask = fft_freq > 0
        fft_freq = fft_freq[positive_freq_mask]
        fft_power = np.abs(fft_result[positive_freq_mask])
        
        # Find peak frequency (natural frequency)
        peak_idx = np.argmax(fft_power)
        f0 = fft_freq[peak_idx]
        
        if visualize:
            plt.figure(figsize=(10, 6))
            plt.plot(fft_freq, fft_power)
            plt.axvline(f0, color='r', linestyle='--', label=f'f0 = {f0:.3f} Hz')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')
            plt.title('Frequency Domain Signal')
            plt.legend()
            plt.grid(True)
            plt.show()
            
        return f0, fft_freq, fft_power
    
    def estimate_damping_ratio(self, fft_freq, fft_power, f0, visualize=False):
        """
        Estimate damping ratio using half-power bandwidth method.
        
        Args:
            fft_freq: Frequency array from FFT
            fft_power: Power spectrum from FFT
            f0: Natural frequency (Hz)
            visualize: Whether to plot the half-power points
            
        Returns:
            zeta0: Damping ratio
        """
        # Find power at natural frequency
        f0_idx = np.argmin(np.abs(fft_freq - f0))
        P_f0 = fft_power[f0_idx]
        
        # Calculate half-power level
        half_power = P_f0 / np.sqrt(2)
        
        # Create interpolation function for more accurate half-power point detection
        interp_func = interp1d(fft_freq, fft_power, kind='cubic', 
                               bounds_error=False, fill_value=0)
        
        # Fine frequency grid around f0
        freq_range = 0.5 * f0  # Search within ±50% of f0
        fine_freq = np.linspace(max(0, f0 - freq_range), 
                                f0 + freq_range, 1000)
        fine_power = interp_func(fine_freq)
        
        # Find frequencies where power crosses half-power level
        # Left side (fl)
        left_mask = fine_freq < f0
        left_freq = fine_freq[left_mask]
        left_power = fine_power[left_mask]
        
        if len(left_freq) > 0:
            fl_idx = np.argmin(np.abs(left_power - half_power))
            fl = left_freq[fl_idx]
        else:
            fl = f0 * 0.95  # Fallback
        
        # Right side (fr)
        right_mask = fine_freq > f0
        right_freq = fine_freq[right_mask]
        right_power = fine_power[right_mask]
        
        if len(right_freq) > 0:
            fr_idx = np.argmin(np.abs(right_power - half_power))
            fr = right_freq[fr_idx]
        else:
            fr = f0 * 1.05  # Fallback
        
        # Calculate damping ratio using Equation (2)
        zeta0 = (fr - fl) / (2 * f0)
        
        if visualize:
            plt.figure(figsize=(10, 6))
            plt.plot(fft_freq, fft_power, 'b-', label='Power Spectrum')
            plt.axhline(half_power, color='g', linestyle='--', 
                       label=f'Half-power level')
            plt.axvline(f0, color='r', linestyle='--', label=f'f0 = {f0:.3f} Hz')
            plt.axvline(fl, color='orange', linestyle=':', label=f'fl = {fl:.3f} Hz')
            plt.axvline(fr, color='orange', linestyle=':', label=f'fr = {fr:.3f} Hz')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')
            plt.title(f'Half-Power Bandwidth Method (ζ0 = {zeta0:.4f})')
            plt.legend()
            plt.grid(True)
            plt.xlim(max(0, f0 - freq_range), f0 + freq_range)
            plt.show()
            
        return zeta0
    
    def analyze_branch(self, parent_markers, child_markers, visualize=False):
        """
        Complete analysis pipeline for a branch.
        
        Args:
            parent_markers: Parent branch marker positions (n_frames, 3)
            child_markers: Child branch marker positions (n_frames, 3)
            visualize: Whether to create visualizations
            
        Returns:
            results: Dictionary containing f0, zeta0, and intermediate data
        """
        # Calculate deflection angle
        #theta_t = self.calculate_deflection_angle(parent_markers, child_markers)        
        theta_t = self.pca_transform(parent_markers, child_markers)
        # Estimate natural frequency
        f0, fft_freq, fft_power = self.estimate_natural_frequency(
            theta_t, visualize=visualize
        )
         
        # Estimate damping ratio
        zeta0 = self.estimate_damping_ratio(
            fft_freq, fft_power, f0, visualize=visualize
        )
        
        results = {
            'natural_frequency': f0,
            'damping_ratio': zeta0,
            'deflection_angle': theta_t,
            'fft_freq': fft_freq,
            'fft_power': fft_power
        }
        
        return results


def plot_markers_3d(parent_markers, child_markers, t=None, show_distance=True):
    """
    Plot 3D trajectories of parent and child markers with equal axis scaling.
    
    Parameters:
    -----------
    parent_markers : np.ndarray
        Array of shape (n_frames, 3) containing parent marker positions
    child_markers : np.ndarray
        Array of shape (n_frames, 3) containing child marker positions
    t : np.ndarray, optional
        Time array of shape (n_frames,). If None, uses frame indices
    show_distance : bool, optional
        Whether to show distance plot (default: True)
    """
    n_frames = len(parent_markers)
    
    if t is None:
        t = np.arange(n_frames)
    
    # Combine all data to find global bounds
    all_data = np.vstack([parent_markers, child_markers])
    max_range = np.array([all_data[:, 0].max() - all_data[:, 0].min(),
                          all_data[:, 1].max() - all_data[:, 1].min(),
                          all_data[:, 2].max() - all_data[:, 2].min()]).max() / 2.0
    
    mid_x = (all_data[:, 0].max() + all_data[:, 0].min()) * 0.5
    mid_y = (all_data[:, 1].max() + all_data[:, 1].min()) * 0.5
    mid_z = (all_data[:, 2].max() + all_data[:, 2].min()) * 0.5
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 6))
    
    # Plot 1: Trajectories
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(parent_markers[:, 0], parent_markers[:, 1], parent_markers[:, 2], 
             'b-', linewidth=1, alpha=0.6, label='Parent Marker')
    ax1.plot(child_markers[:, 0], child_markers[:, 1], child_markers[:, 2], 
             'r-', linewidth=1, alpha=0.6, label='Child Marker')
    ax1.scatter(parent_markers[0, 0], parent_markers[0, 1], parent_markers[0, 2], 
               c='blue', s=100, marker='o', label='Start (Parent)')
    ax1.scatter(child_markers[0, 0], child_markers[0, 1], child_markers[0, 2], 
               c='red', s=100, marker='o', label='Start (Child)')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_zlabel('Z Position (m)')
    ax1.set_title('3D Marker Trajectories')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Set equal axis limits
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)
    ax1.set_box_aspect([1, 1, 1])
    
    # Plot 2: Time-colored trajectory for child marker
    ax2 = fig.add_subplot(122, projection='3d')
    scatter = ax2.scatter(child_markers[:, 0], child_markers[:, 1], child_markers[:, 2], 
                         c=t, cmap='viridis', s=10, alpha=0.6)
    ax2.plot(parent_markers[:, 0], parent_markers[:, 1], parent_markers[:, 2], 
             'b-', linewidth=2, alpha=0.8, label='Parent Marker')
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_zlabel('Z Position (m)')
    ax2.set_title('Child Marker Trajectory (Time-Colored)')
    cbar = plt.colorbar(scatter, ax=ax2, pad=0.1, shrink=0.8)
    cbar.set_label('Time (s)' if t is not None else 'Frame')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Set equal axis limits
    ax2.set_xlim(mid_x - max_range, mid_x + max_range)
    ax2.set_ylim(mid_y - max_range, mid_y + max_range)
    ax2.set_zlim(mid_z - max_range, mid_z + max_range)
    ax2.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    plt.show()
    
    # Additional plot: Distance between markers over time
    if show_distance:
        fig2, ax = plt.subplots(figsize=(10, 4))
        distance = np.linalg.norm(child_markers - parent_markers, axis=1)
        ax.plot(t, distance, 'g-', linewidth=2)
        ax.set_xlabel('Time (s)' if t is not None else 'Frame')
        ax.set_ylabel('Distance (m)')
        ax.set_title('Distance Between Parent and Child Markers')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def animate_markers_3d(parent_markers, child_markers, t=None, 
                       trail_length=100, fps=30, realtime=True, save_path=None):
    """
    Create an animated 3D plot of parent and child markers.
    
    Parameters:
    -----------
    parent_markers : np.ndarray
        Array of shape (n_frames, 3) containing parent marker positions
    child_markers : np.ndarray
        Array of shape (n_frames, 3) containing child marker positions
    t : np.ndarray, optional
        Time array of shape (n_frames,). If None, uses frame indices
    trail_length : int, optional
        Number of past frames to show as trail (default: 100)
    fps : int, optional
        Target frames per second for animation (default: 30)
    realtime : bool, optional
        If True, subsample data to match real-time playback (default: True)
    save_path : str, optional
        If provided, save animation to this path (e.g., 'animation.gif')
    
    Returns:
    --------
    animation : FuncAnimation object
    """
    n_frames = len(parent_markers)
    
    if t is None:
        t = np.arange(n_frames)
    
    # Calculate sampling rate from time array
    if len(t) > 1:
        original_sampling_rate = 1.0 / (t[1] - t[0])
    else:
        original_sampling_rate = 100  # Default assumption
    
    # Subsample data for real-time playback
    if realtime and original_sampling_rate > fps:
        skip = int(original_sampling_rate / fps)
        frame_indices = np.arange(0, n_frames, skip)
        parent_markers_sub = parent_markers[frame_indices]
        child_markers_sub = child_markers[frame_indices]
        t_sub = t[frame_indices]
        trail_length_sub = max(1, trail_length // skip)
        print(f"Original: {n_frames} frames at {original_sampling_rate:.1f} Hz")
        print(f"Subsampled: {len(frame_indices)} frames at {fps} fps (every {skip} frames)")
    else:
        frame_indices = np.arange(n_frames)
        parent_markers_sub = parent_markers
        child_markers_sub = child_markers
        t_sub = t
        trail_length_sub = trail_length
    
    n_frames_sub = len(frame_indices)
    
    # Calculate equal axis bounds
    all_data = np.vstack([parent_markers, child_markers])
    max_range = np.array([all_data[:, 0].max() - all_data[:, 0].min(),
                          all_data[:, 1].max() - all_data[:, 1].min(),
                          all_data[:, 2].max() - all_data[:, 2].min()]).max() / 2.0
    
    mid_x = (all_data[:, 0].max() + all_data[:, 0].min()) * 0.5
    mid_y = (all_data[:, 1].max() + all_data[:, 1].min()) * 0.5
    mid_z = (all_data[:, 2].max() + all_data[:, 2].min()) * 0.5
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize plot elements
    parent_point, = ax.plot([], [], [], 'bo', markersize=10, label='Parent Marker')
    child_point, = ax.plot([], [], [], 'ro', markersize=10, label='Child Marker')
    parent_trail, = ax.plot([], [], [], 'b-', linewidth=1, alpha=0.3)
    child_trail, = ax.plot([], [], [], 'r-', linewidth=1, alpha=0.3)
    connection_line, = ax.plot([], [], [], 'g--', linewidth=1, alpha=0.5)
    
    # Time text
    time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
    
    # Set up the plot
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('3D Marker Animation (Real-time)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Set equal axis limits
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_box_aspect([1, 1, 1])
    
    def init():
        parent_point.set_data([], [])
        parent_point.set_3d_properties([])
        child_point.set_data([], [])
        child_point.set_3d_properties([])
        parent_trail.set_data([], [])
        parent_trail.set_3d_properties([])
        child_trail.set_data([], [])
        child_trail.set_3d_properties([])
        connection_line.set_data([], [])
        connection_line.set_3d_properties([])
        time_text.set_text('')
        return parent_point, child_point, parent_trail, child_trail, connection_line, time_text
    
    def update(frame):
        # Current positions
        parent_point.set_data([parent_markers_sub[frame, 0]], [parent_markers_sub[frame, 1]])
        parent_point.set_3d_properties([parent_markers_sub[frame, 2]])
        
        child_point.set_data([child_markers_sub[frame, 0]], [child_markers_sub[frame, 1]])
        child_point.set_3d_properties([child_markers_sub[frame, 2]])
        
        # Trail (last trail_length frames)
        start_idx = max(0, frame - trail_length_sub)
        parent_trail.set_data(parent_markers_sub[start_idx:frame+1, 0], 
                             parent_markers_sub[start_idx:frame+1, 1])
        parent_trail.set_3d_properties(parent_markers_sub[start_idx:frame+1, 2])
        
        child_trail.set_data(child_markers_sub[start_idx:frame+1, 0], 
                            child_markers_sub[start_idx:frame+1, 1])
        child_trail.set_3d_properties(child_markers_sub[start_idx:frame+1, 2])
        
        # Connection line
        connection_line.set_data([parent_markers_sub[frame, 0], child_markers_sub[frame, 0]], 
                                [parent_markers_sub[frame, 1], child_markers_sub[frame, 1]])
        connection_line.set_3d_properties([parent_markers_sub[frame, 2], child_markers_sub[frame, 2]])
        
        # Update time text
        time_text.set_text(f'Time: {t_sub[frame]:.2f} s')
        
        return parent_point, child_point, parent_trail, child_trail, connection_line, time_text
    
    # Create animation with interval matching fps
    interval = 1000 / fps  # milliseconds per frame
    anim = FuncAnimation(fig, update, frames=n_frames_sub, init_func=init,
                        blit=True, interval=interval, repeat=True)
    
    # Save if path provided
    if save_path:
        print(f"Saving animation to {save_path}...")
        if save_path.endswith('.gif'):
            anim.save(save_path, writer=PillowWriter(fps=fps))
        else:
            anim.save(save_path, writer='ffmpeg', fps=fps)
        print("Animation saved!")
    
    plt.show()
    
    return anim

# Example usage
if __name__ == "__main__":
    # Simulate example data (replace with your actual marker data)
    sampling_rate = 100  # Hz
    duration = 10  # seconds
    n_frames = int(sampling_rate * duration)
    t = np.linspace(0, duration, n_frames)
    
    # Simulate damped oscillation for demonstration
    f_natural = 2.5  # Hz
    zeta = 0.05  # Damping ratio
    omega_d = 2 * np.pi * f_natural * np.sqrt(1 - zeta**2)
    
    # Simulated parent marker (relatively stationary)
    parent_markers = np.zeros((n_frames, 3))
    parent_markers[:, 0] = 0.1 * np.random.randn(n_frames) * 0.01  # Small noise
    
    # Simulated child marker (oscillating)
    amplitude = 0.5
    decay = np.exp(-zeta * 2 * np.pi * f_natural * t)
    oscillation = amplitude * decay * np.sin(omega_d * t)
    
    child_markers = np.zeros((n_frames, 3))
    child_markers[:, 0] = 1.0 + oscillation * 0.1
    child_markers[:, 1] = oscillation
    child_markers[:, 2] = 0.05 * np.random.randn(n_frames) * 0.01

    # Create animation (optionally save as GIF)
    # anim = animate_markers_3d(parent_markers, child_markers, t, 
    #                           trail_length=100, fps=30, realtime=True)

    plot_markers_3d(parent_markers, child_markers, t)

    # Create estimator and analyze
    estimator = BranchPropertyEstimator(sampling_rate)
    results = estimator.analyze_branch(
        parent_markers, 
        child_markers, 
        visualize=True
    )
    
    print(f"\nEstimated Natural Frequency: {results['natural_frequency']:.3f} Hz")
    print(f"Actual Natural Frequency: {f_natural:.3f} Hz")
    print(f"\nEstimated Damping Ratio: {results['damping_ratio']:.4f}")
    print(f"Actual Damping Ratio: {zeta:.4f}")