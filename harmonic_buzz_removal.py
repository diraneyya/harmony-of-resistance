import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm

class HarmonicBuzzRemover:
    def __init__(self, noise_sample_path):
        """
        Initialize with a clean sample of the buzz noise to be removed
        """
        self.noise_audio, self.sr = librosa.load(noise_sample_path, sr=None)
        self.duration = len(self.noise_audio) / self.sr
        print(f"Loaded noise sample: {self.duration:.2f}s at {self.sr}Hz")
        
        # STFT parameters
        self.n_fft = 2048
        self.hop_length = 512
        self.window = 'hann'
        
        # Analysis results
        self.noise_frequencies = None
        self.noise_amplitudes = None
        self.noise_profile = None
        
    def analyze_noise_profile(self, plot=True):
        """
        Analyze the harmonic components of the noise sample
        """
        print("Analyzing noise harmonic profile...")
        
        # Perform STFT on noise sample
        stft = librosa.stft(self.noise_audio, 
                           n_fft=self.n_fft, 
                           hop_length=self.hop_length, 
                           window=self.window)
        
        # Get average magnitude spectrum
        magnitude = np.abs(stft)
        avg_magnitude = np.mean(magnitude, axis=1)
        
        # Find peaks in the spectrum (harmonic components)
        peaks, properties = signal.find_peaks(avg_magnitude, 
                                            height=np.max(avg_magnitude)*0.05,
                                            distance=10)
        
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        self.noise_frequencies = freqs[peaks]
        self.noise_amplitudes = avg_magnitude[peaks]
        
        # Store the complete noise profile for spectral subtraction
        self.noise_profile = avg_magnitude
        
        print(f"Found {len(self.noise_frequencies)} dominant noise frequencies")
        
        if plot:
            self.plot_noise_analysis(freqs, avg_magnitude)
        
        # Print dominant frequencies
        sorted_peaks = sorted(zip(self.noise_frequencies, self.noise_amplitudes), 
                            key=lambda x: x[1], reverse=True)
        print("\nTop 10 Noise Frequencies:")
        for i, (freq, amp) in enumerate(sorted_peaks[:10]):
            print(f"{i+1}: {freq:.1f} Hz (amplitude: {amp:.3f})")
        
        return self.noise_frequencies, self.noise_amplitudes
    
    def plot_noise_analysis(self, freqs, avg_magnitude):
        """
        Plot the noise analysis results
        """
        plt.ioff()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Full spectrum
        ax1.plot(freqs[:len(freqs)//2], avg_magnitude[:len(freqs)//2])
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Magnitude')
        ax1.set_title('Noise Spectrum')
        ax1.set_xlim(0, self.sr//2)
        
        # Dominant frequencies
        peaks, _ = signal.find_peaks(avg_magnitude, height=np.max(avg_magnitude)*0.05, distance=10)
        peak_freqs = freqs[peaks]
        peak_mags = avg_magnitude[peaks]
        
        ax2.stem(peak_freqs, peak_mags, basefmt=' ')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude')
        ax2.set_title('Dominant Noise Components')
        ax2.set_xlim(0, min(5000, self.sr//2))
        
        plt.tight_layout()
        plt.savefig('noise_analysis.png', dpi=150, bbox_inches='tight')
        print("Noise analysis plot saved as 'noise_analysis.png'")
        plt.close(fig)
        plt.ion()
    
    def remove_noise_from_audio(self, song_path, output_path, method='spectral_subtraction', 
                               noise_reduction_factor=1.0, spectral_floor=0.1):
        """
        Remove the analyzed noise from a song - STEREO VERSION
        """
        print(f"Loading song: {song_path}")
        
        # Load as STEREO (mono=False is crucial)
        song_audio, song_sr = librosa.load(song_path, sr=None, mono=False)
        
        # Handle both mono and stereo inputs
        if song_audio.ndim == 1:
            print("Input is mono, converting to stereo")
            song_audio = np.array([song_audio, song_audio])
        
        song_duration = song_audio.shape[1] / song_sr
        print(f"Song duration: {song_duration:.2f}s at {song_sr}Hz, {song_audio.shape[0]} channels")
        
        if self.noise_frequencies is None:
            self.analyze_noise_profile(plot=False)
        
        if method == 'spectral_subtraction':
            result = self._spectral_subtraction_method_stereo(song_audio, song_sr, 
                                                            noise_reduction_factor, spectral_floor)
        else:
            raise ValueError("Only spectral_subtraction method supported for stereo")
        
        # Save the result as stereo
        sf.write(output_path, result.T, song_sr)  # Transpose for soundfile format
        print(f"Stereo cleaned audio saved to: {output_path}")
        
        return result
    
    def _spectral_subtraction_method_stereo(self, song_audio, song_sr, reduction_factor, spectral_floor):
        """
        Subtract noise spectrum from song spectrum - STEREO VERSION
        """
        print("Applying spectral subtraction to stereo audio...")
        
        # Resample noise profile to match song sample rate if needed
        if song_sr != self.sr:
            freq_scale = song_sr / self.sr
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
            cleaned_channel = self._process_single_channel(channel_audio, song_sr, 
                                                         noise_profile_resampled, 
                                                         reduction_factor, spectral_floor)
            cleaned_channels.append(cleaned_channel)
        
        # Combine channels back to stereo array
        result = np.array(cleaned_channels)
        return result
    
    def _process_single_channel(self, channel_audio, song_sr, noise_profile_resampled, 
                               reduction_factor, spectral_floor):
        """
        Process a single channel using the exact same method as before
        """
        # Process in chunks to handle long audio files
        chunk_duration = 10.0  # Process 10 seconds at a time
        chunk_samples = int(chunk_duration * song_sr)
        num_chunks = int(np.ceil(len(channel_audio) / chunk_samples))
        
        result = np.zeros_like(channel_audio)
        
        for i in tqdm(range(num_chunks), desc="Processing chunks", leave=False):
            start_idx = i * chunk_samples
            end_idx = min(start_idx + chunk_samples, len(channel_audio))
            chunk = channel_audio[start_idx:end_idx]
            
            # STFT of song chunk - EXACT SAME AS BEFORE
            song_stft = librosa.stft(chunk, n_fft=self.n_fft, 
                                   hop_length=self.hop_length, window=self.window)
            song_magnitude = np.abs(song_stft)
            song_phase = np.angle(song_stft)
            
            # Subtract noise profile - EXACT SAME AS BEFORE
            noise_mag = noise_profile_resampled[:song_magnitude.shape[0], np.newaxis]
            cleaned_magnitude = song_magnitude - reduction_factor * noise_mag
            
            # Apply spectral floor to prevent artifacts - EXACT SAME AS BEFORE
            cleaned_magnitude = np.maximum(cleaned_magnitude, 
                                         spectral_floor * song_magnitude)
            
            # Reconstruct audio - EXACT SAME AS BEFORE
            cleaned_stft = cleaned_magnitude * np.exp(1j * song_phase)
            chunk_result = librosa.istft(cleaned_stft, hop_length=self.hop_length, 
                                       window=self.window)
            
            # Add to result - EXACT SAME AS BEFORE
            result[start_idx:start_idx + len(chunk_result)] = chunk_result
        
        return result

# Example usage
if __name__ == "__main__":
    # Initialize with your noise sample
    remover = HarmonicBuzzRemover("zannana.mp3")
    
    # Analyze the noise profile
    remover.analyze_noise_profile(plot=True)
    
    # Remove noise from song using your best parameters
    print("\n" + "="*50)
    print("Removing noise from song.mp3 (STEREO VERSION)...")
    
    # Your best parameters: reduction factor 1.0, spectral floor 0.01
    print("\n=== Spectral Subtraction Method (Stereo) ===")
    remover.remove_noise_from_audio("song.mp3", 
                                   "song_cleaned_stereo.wav",
                                   method='spectral_subtraction',
                                   noise_reduction_factor=1.0,
                                   spectral_floor=0.01)
    
    print("\n" + "="*50)
    print("Stereo noise removal complete!")
    print("Generated file: song_cleaned_stereo.wav")
    print("This uses the exact same noise removal method but preserves stereo!")