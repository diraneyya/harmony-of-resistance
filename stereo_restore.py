import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from tqdm import tqdm

class StereoRestorationTool:
    def __init__(self, drone_frequencies):
        """
        Initialize with known drone frequencies to target for restoration
        """
        self.drone_frequencies = drone_frequencies
        self.n_fft = 2048
        self.hop_length = 512
        self.window = 'hann'
        
    def restore_stereo_spatial_info(self, mono_cleaned_path, original_stereo_path, output_path):
        """
        Restore stereo spatial information from original, but keep mono cleaning for drone frequencies
        """
        print("Loading audio files...")
        
        # Load mono cleaned version (your working version)
        mono_cleaned, sr = librosa.load(mono_cleaned_path, sr=None, mono=True)
        
        # Load original stereo
        stereo_original, sr_orig = librosa.load(original_stereo_path, sr=None, mono=False)
        
        # Ensure sample rates match
        if sr != sr_orig:
            print(f"Resampling stereo from {sr_orig}Hz to {sr}Hz")
            stereo_original = librosa.resample(stereo_original, orig_sr=sr_orig, target_sr=sr)
        
        # Handle stereo format
        if stereo_original.ndim == 1:
            stereo_original = np.array([stereo_original, stereo_original])
        
        # Ensure same length
        min_length = min(len(mono_cleaned), stereo_original.shape[1])
        mono_cleaned = mono_cleaned[:min_length]
        stereo_original = stereo_original[:, :min_length]
        
        print("Processing frequency-selective restoration...")
        
        # Process in chunks
        chunk_duration = 5.0
        chunk_samples = int(chunk_duration * sr)
        num_chunks = int(np.ceil(min_length / chunk_samples))
        
        restored_stereo = np.zeros((2, min_length))
        
        for i in tqdm(range(num_chunks), desc="Restoring chunks"):
            start_idx = i * chunk_samples
            end_idx = min(start_idx + chunk_samples, min_length)
            
            # Get chunks
            mono_chunk = mono_cleaned[start_idx:end_idx]
            stereo_chunk = stereo_original[:, start_idx:end_idx]
            
            # Restore this chunk
            restored_chunk = self._restore_chunk_frequency_selective(mono_chunk, stereo_chunk, sr)
            
            # Store result
            restored_stereo[:, start_idx:start_idx + restored_chunk.shape[1]] = restored_chunk
        
        # Save result
        sf.write(output_path, restored_stereo.T, sr)
        print(f"Stereo-restored audio saved to: {output_path}")
        
        return restored_stereo
    
    def _restore_chunk_frequency_selective(self, mono_chunk, stereo_chunk, sr):
        """
        Restore stereo info by frequency bands - use mono for drone frequencies, stereo for others
        """
        # Pad chunks to same length if needed
        if len(mono_chunk) != stereo_chunk.shape[1]:
            target_length = max(len(mono_chunk), stereo_chunk.shape[1])
            if len(mono_chunk) < target_length:
                mono_chunk = np.pad(mono_chunk, (0, target_length - len(mono_chunk)))
            if stereo_chunk.shape[1] < target_length:
                stereo_chunk = np.pad(stereo_chunk, ((0, 0), (0, target_length - stereo_chunk.shape[1])))
        
        # STFT of all channels
        mono_stft = librosa.stft(mono_chunk, n_fft=self.n_fft, hop_length=self.hop_length)
        left_stft = librosa.stft(stereo_chunk[0], n_fft=self.n_fft, hop_length=self.hop_length)
        right_stft = librosa.stft(stereo_chunk[1], n_fft=self.n_fft, hop_length=self.hop_length)
        
        # Get frequency bins
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        
        # Create frequency mask for drone frequencies
        drone_mask = np.zeros(len(freqs), dtype=bool)
        
        for drone_freq in self.drone_frequencies:
            # Find frequency bins around each drone frequency (±20 Hz tolerance)
            freq_tolerance = 20  # Hz
            freq_mask = (freqs >= drone_freq - freq_tolerance) & (freqs <= drone_freq + freq_tolerance)
            drone_mask |= freq_mask
        
        # Create output STFTs
        output_left_stft = left_stft.copy()
        output_right_stft = right_stft.copy()
        
        # Replace drone frequency regions with mono cleaned version
        for freq_idx in range(len(freqs)):
            if drone_mask[freq_idx]:
                # Use cleaned mono version for drone frequencies
                output_left_stft[freq_idx, :] = mono_stft[freq_idx, :]
                output_right_stft[freq_idx, :] = mono_stft[freq_idx, :]
            # else: keep original stereo content for non-drone frequencies
        
        # Convert back to time domain
        restored_left = librosa.istft(output_left_stft, hop_length=self.hop_length)
        restored_right = librosa.istft(output_right_stft, hop_length=self.hop_length)
        
        return np.array([restored_left, restored_right])
    
    def create_frequency_mask_visualization(self, sr, save_path='frequency_mask.png'):
        """
        Visualize which frequencies will be replaced vs preserved
        """
        import matplotlib.pyplot as plt
        
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        
        # Create mask
        drone_mask = np.zeros(len(freqs), dtype=bool)
        for drone_freq in self.drone_frequencies:
            freq_tolerance = 20
            freq_mask = (freqs >= drone_freq - freq_tolerance) & (freqs <= drone_freq + freq_tolerance)
            drone_mask |= freq_mask
        
        plt.figure(figsize=(12, 6))
        
        # Plot frequency response
        y_mono = np.where(drone_mask[:len(freqs)//2], 1, 0)
        y_stereo = np.where(~drone_mask[:len(freqs)//2], 1, 0)
        
        plt.fill_between(freqs[:len(freqs)//2], 0, y_mono, alpha=0.7, color='red', label='Mono cleaned (drone frequencies)')
        plt.fill_between(freqs[:len(freqs)//2], 0, y_stereo, alpha=0.7, color='blue', label='Original stereo (music frequencies)')
        
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Source')
        plt.title('Frequency-Selective Stereo Restoration Map')
        plt.legend()
        plt.xlim(0, 2000)  # Focus on relevant frequency range
        plt.grid(True, alpha=0.3)
        
        # Mark drone frequencies
        for freq in self.drone_frequencies:
            if freq < 2000:
                plt.axvline(freq, color='red', linestyle='--', alpha=0.8)
                plt.text(freq, 0.5, f'{freq:.0f}Hz', rotation=90, ha='right')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Frequency mask visualization saved to: {save_path}")

# Alternative simpler approach
class SimpleMonoToStereoRestorer:
    def __init__(self):
        pass
    
    def restore_using_side_information(self, mono_cleaned_path, original_stereo_path, output_path):
        """
        Simpler approach: Use stereo side information to restore spatial width
        """
        print("Loading files for side information restoration...")
        
        # Load cleaned mono
        mono_cleaned, sr = librosa.load(mono_cleaned_path, sr=None, mono=True)
        
        # Load original stereo
        stereo_original, sr_orig = librosa.load(original_stereo_path, sr=None, mono=False)
        
        if sr != sr_orig:
            stereo_original = librosa.resample(stereo_original, orig_sr=sr_orig, target_sr=sr)
        
        if stereo_original.ndim == 1:
            stereo_original = np.array([stereo_original, stereo_original])
        
        # Ensure same length
        min_length = min(len(mono_cleaned), stereo_original.shape[1])
        mono_cleaned = mono_cleaned[:min_length]
        stereo_original = stereo_original[:, :min_length]
        
        # Calculate original stereo side information
        original_mid = (stereo_original[0] + stereo_original[1]) / 2  # Mid (mono sum)
        original_side = (stereo_original[0] - stereo_original[1]) / 2  # Side (stereo difference)
        
        # High-pass filter the side information to preserve only high-frequency stereo content
        # This removes low-frequency drone content from the side channel
        sos = signal.butter(4, 500, btype='high', fs=sr, output='sos')  # 500 Hz high-pass
        filtered_side = signal.sosfilt(sos, original_side)
        
        # Create new stereo from cleaned mono + filtered side
        restored_left = mono_cleaned + filtered_side
        restored_right = mono_cleaned - filtered_side
        
        # Save result
        restored_stereo = np.array([restored_left, restored_right])
        sf.write(output_path, restored_stereo.T, sr)
        print(f"Side-information restored audio saved to: {output_path}")
        
        return restored_stereo

# Example usage
if __name__ == "__main__":
    # Known drone frequencies from your analysis
    drone_frequencies = [344.5, 581.4, 925.9, 107.7, 1399.7, 1162.8]
    
    print("STEREO SPATIAL RESTORATION")
    print("="*50)
    
    # Method 1: Frequency-selective restoration
    print("\n=== Method 1: Frequency-Selective Restoration ===")
    restorer = StereoRestorationTool(drone_frequencies)
    
    # Create visualization
    restorer.create_frequency_mask_visualization(44100, 'frequency_restoration_map.png')
    
    # Restore stereo spatial info
    restorer.restore_stereo_spatial_info(
        "song_cleaned_spectral.wav",  # Your working mono cleaned version
        "song.mp3",                   # Original stereo
        "song_stereo_frequency_restored.wav"
    )
    
    # Method 2: Side information restoration (simpler)
    print("\n=== Method 2: Side Information Restoration ===")
    simple_restorer = SimpleMonoToStereoRestorer()
    
    simple_restorer.restore_using_side_information(
        "song_cleaned_spectral.wav",  # Your working mono cleaned version
        "song.mp3",                   # Original stereo
        "song_stereo_side_restored.wav"
    )
    
    print("\n" + "="*50)
    print("RESTORATION COMPLETE!")
    print("Generated files:")
    print("- song_stereo_frequency_restored.wav (frequency-selective method)")
    print("- song_stereo_side_restored.wav (side information method)")
    print("- frequency_restoration_map.png (visualization)")
    print("\nTry both methods to see which preserves stereo better while keeping clean drone removal.")