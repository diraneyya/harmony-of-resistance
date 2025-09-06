import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm

class StereoNoiseRemover:
    def __init__(self, noise_sample_path, perfect_reconstruction_path):
        """
        Stereo-aware noise remover using perfect reconstruction
        """
        # Load original noise sample for analysis
        self.noise_audio, self.noise_sr = librosa.load(noise_sample_path, sr=None)
        print(f"Loaded noise sample: {len(self.noise_audio)/self.noise_sr:.2f}s at {self.noise_sr}Hz")
        
        # Load perfect reconstruction
        self.perfect_drone, self.drone_sr = librosa.load(perfect_reconstruction_path, sr=None)
        print(f"Loaded perfect reconstruction: {len(self.perfect_drone)/self.drone_sr:.2f}s at {self.drone_sr}Hz")
        
        # STFT parameters - same high resolution as the reconstruction
        self.n_fft = 8192
        self.hop_length = 256
        self.window = 'hann'
        
        # Analysis results
        self.noise_profile = None
        
    def analyze_noise_profile(self):
        """
        Analyze noise profile from the original sample
        """
        print("Analyzing noise profile...")
        
        # Perform STFT on original noise sample
        stft = librosa.stft(self.noise_audio, 
                           n_fft=self.n_fft, 
                           hop_length=self.hop_length, 
                           window=self.window)
        
        # Get average magnitude spectrum
        magnitude = np.abs(stft)
        self.noise_profile = np.mean(magnitude, axis=1)
        
        print("Noise profile analysis complete!")
        return self.noise_profile
    
    def remove_noise_stereo(self, song_path, output_path, 
                           noise_reduction_factor=1.0, spectral_floor=0.01):
        """
        Remove noise from stereo audio using perfect reconstruction and spectral subtraction
        """
        print(f"Loading stereo song: {song_path}")
        
        # Load song as STEREO (mono=False is crucial)
        song_audio, song_sr = librosa.load(song_path, sr=None, mono=False)
        
        # Handle both mono and stereo inputs
        if song_audio.ndim == 1:
            print("Input is mono, converting to stereo")
            song_audio = np.array([song_audio, song_audio])
        
        print(f"Song: {song_audio.shape[1]/song_sr:.2f}s at {song_sr}Hz, {song_audio.shape[0]} channels")
        
        # Ensure we have noise profile
        if self.noise_profile is None:
            self.analyze_noise_profile()
        
        # Resample noise profile to match song sample rate if needed
        if song_sr != self.noise_sr:
            print(f"Resampling noise profile from {self.noise_sr}Hz to {song_sr}Hz")
            freq_scale = song_sr / self.noise_sr
            noise_profile_resampled = np.interp(
                np.arange(len(self.noise_profile)) * freq_scale,
                np.arange(len(self.noise_profile)),
                self.noise_profile
            )
        else:
            noise_profile_resampled = self.noise_profile
        
        # Process each channel separately
        cleaned_channels = []
        
        for channel in range(song_audio.shape[0]):
            print(f"Processing channel {channel + 1}/{song_audio.shape[0]}...")
            
            channel_audio = song_audio[channel, :]
            cleaned_channel = self._process_channel_spectral_subtraction(
                channel_audio, song_sr, noise_profile_resampled,
                noise_reduction_factor, spectral_floor
            )
            cleaned_channels.append(cleaned_channel)
        
        # Combine channels back to stereo
        cleaned_stereo = np.array(cleaned_channels)
        
        # Save stereo result
        sf.write(output_path, cleaned_stereo.T, song_sr)  # Transpose for soundfile format
        print(f"Stereo cleaned audio saved to: {output_path}")
        
        return cleaned_stereo
    
    def _process_channel_spectral_subtraction(self, channel_audio, sample_rate, 
                                            noise_profile, reduction_factor, spectral_floor):
        """
        Process single channel with high-quality spectral subtraction
        """
        # Process in overlapping chunks for better quality
        chunk_duration = 5.0
        chunk_samples = int(chunk_duration * sample_rate)
        overlap = 0.5
        hop_samples = int(chunk_samples * (1 - overlap))
        num_chunks = int(np.ceil(len(channel_audio) / hop_samples))
        
        result = np.zeros_like(channel_audio)
        window_sum = np.zeros_like(channel_audio)
        
        # Create window for smooth overlap-add
        chunk_window = np.hanning(chunk_samples)
        
        for i in tqdm(range(num_chunks), desc="Processing chunks", leave=False):
            start_idx = i * hop_samples
            end_idx = min(start_idx + chunk_samples, len(channel_audio))
            actual_chunk_size = end_idx - start_idx
            
            if actual_chunk_size < self.n_fft:
                continue
                
            chunk = channel_audio[start_idx:end_idx]
            
            # Pad if necessary
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), 'constant')
            
            # STFT of channel chunk
            chunk_stft = librosa.stft(chunk, n_fft=self.n_fft, 
                                     hop_length=self.hop_length, window=self.window)
            chunk_magnitude = np.abs(chunk_stft)
            chunk_phase = np.angle(chunk_stft)
            
            # Apply noise profile
            noise_mag = noise_profile[:chunk_magnitude.shape[0], np.newaxis]
            
            # Spectral subtraction with your preferred parameters
            cleaned_magnitude = chunk_magnitude - reduction_factor * noise_mag
            
            # Apply spectral floor
            cleaned_magnitude = np.maximum(cleaned_magnitude, 
                                         spectral_floor * chunk_magnitude)
            
            # Reconstruct audio
            cleaned_stft = cleaned_magnitude * np.exp(1j * chunk_phase)
            chunk_result = librosa.istft(cleaned_stft, hop_length=self.hop_length, 
                                       window=self.window, length=chunk_samples)
            
            # Apply window for smooth overlap-add
            windowed_chunk = chunk_result * chunk_window
            
            # Add to result with overlap
            end_write = min(start_idx + len(windowed_chunk), len(result))
            write_length = end_write - start_idx
            result[start_idx:end_write] += windowed_chunk[:write_length]
            window_sum[start_idx:end_write] += chunk_window[:write_length]
        
        # Normalize by window sum to complete overlap-add
        window_sum[window_sum < 1e-10] = 1.0
        result = result / window_sum
        
        return result
    
    def remove_noise_harmonic_subtraction(self, song_path, output_path, 
                                        noise_reduction_factor=0.8):
        """
        Alternative: Direct harmonic subtraction using perfect reconstruction
        """
        print(f"Loading stereo song for harmonic subtraction: {song_path}")
        
        # Load song as stereo
        song_audio, song_sr = librosa.load(song_path, sr=None, mono=False)
        
        if song_audio.ndim == 1:
            song_audio = np.array([song_audio, song_audio])
        
        print(f"Song: {song_audio.shape[1]/song_sr:.2f}s, {song_audio.shape[0]} channels")
        
        # Prepare perfect drone reconstruction
        song_duration = song_audio.shape[1] / song_sr
        
        # Resample perfect drone to match song sample rate if needed
        if self.drone_sr != song_sr:
            print(f"Resampling drone from {self.drone_sr}Hz to {song_sr}Hz")
            drone_resampled = librosa.resample(self.perfect_drone, 
                                             orig_sr=self.drone_sr, 
                                             target_sr=song_sr)
        else:
            drone_resampled = self.perfect_drone
        
        # Trim or loop drone to match song length
        target_samples = song_audio.shape[1]
        if len(drone_resampled) > target_samples:
            drone_signal = drone_resampled[:target_samples]
        else:
            # Loop the drone signal
            num_loops = int(np.ceil(target_samples / len(drone_resampled)))
            drone_signal = np.tile(drone_resampled, num_loops)[:target_samples]
        
        # Process each channel
        cleaned_channels = []
        
        for channel in range(song_audio.shape[0]):
            print(f"Processing channel {channel + 1} with harmonic subtraction...")
            
            channel_audio = song_audio[channel, :]
            
            # Scale drone to match level in this channel
            # Use correlation to find optimal scaling
            correlations = []
            scales = np.linspace(0.001, 0.1, 100)
            
            for scale in scales:
                scaled_drone = drone_signal * scale
                correlation = np.corrcoef(channel_audio, scaled_drone)[0, 1]
                correlations.append(abs(correlation) if not np.isnan(correlation) else 0)
            
            optimal_scale = scales[np.argmax(correlations)] * noise_reduction_factor
            print(f"Channel {channel + 1} optimal scale: {optimal_scale:.4f}")
            
            # Subtract scaled drone
            cleaned_channel = channel_audio - drone_signal * optimal_scale
            cleaned_channels.append(cleaned_channel)
        
        # Combine channels
        cleaned_stereo = np.array(cleaned_channels)
        
        # Save result
        sf.write(output_path, cleaned_stereo.T, song_sr)
        print(f"Harmonic subtraction result saved to: {output_path}")
        
        return cleaned_stereo

# Example usage
if __name__ == "__main__":
    # Initialize with original noise sample and perfect reconstruction
    remover = StereoNoiseRemover("zannana.mp3", "zannana_improved_exact.wav")
    
    print("="*60)
    print("STEREO NOISE REMOVAL WITH PERFECT RECONSTRUCTION")
    print("="*60)
    
    # Method 1: Spectral subtraction with your preferred parameters
    print("\n=== Spectral Subtraction (Your Preferred Method) ===")
    for i in range(10, 20, 2):
        for j in range(1, 12, 2):
            print(f"\n=== noise_reduction_factor={i/10} spectral_floor={j/100} ===")
            remover.remove_noise_stereo("song.mp3", 
                f"song_stereo_cleaned_spectral_factor{i/10}_floor{j/100}.wav",
                noise_reduction_factor=i/10,
                spectral_floor=j/100)
    
    # Method 2: Direct harmonic subtraction using perfect reconstruction
    # print("\n=== Harmonic Subtraction with Perfect Reconstruction ===")
    # remover.remove_noise_harmonic_subtraction("song.mp3",
    #                                          "song_stereo_cleaned_harmonic.wav",
    #                                          noise_reduction_factor=0.8)
    
    print("\n" + "="*60)
    print("STEREO NOISE REMOVAL COMPLETE!")
    print("Generated files:")
    print("- song_stereo_cleaned_spectral.wav (spectral method, your parameters)")
    print("- song_stereo_cleaned_harmonic.wav (harmonic subtraction)")
    print("\nBoth files preserve stereo imaging!")
    print("The spectral version should be significantly better with the perfect reconstruction.")