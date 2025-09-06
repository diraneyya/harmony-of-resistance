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
    
    def synthesize_noise_signal(self, duration_seconds, target_sr=None):
        """
        Synthesize the noise signal for the given duration
        """
        if self.noise_frequencies is None:
            self.analyze_noise_profile(plot=False)
        
        if target_sr is None:
            target_sr = self.sr
        
        print(f"Synthesizing noise signal for {duration_seconds:.2f}s at {target_sr}Hz...")
        
        # Generate time vector
        t = np.linspace(0, duration_seconds, int(duration_seconds * target_sr))
        
        # Synthesize noise from harmonics
        synthesized = np.zeros_like(t)
        
        for freq, amp in tqdm(zip(self.noise_frequencies, self.noise_amplitudes), 
                             desc="Synthesizing harmonics", 
                             total=len(self.noise_frequencies)):
            if freq > 0:  # Skip DC component
                # Pure steady harmonics
                synthesized += amp * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2*np.pi))
        
        # Normalize to prevent clipping
        if np.max(np.abs(synthesized)) > 0:
            synthesized = synthesized / np.max(np.abs(synthesized)) * 0.9
        
        return synthesized
    
    def remove_noise_from_audio(self, song_path, output_path, method='spectral_subtraction', 
                               noise_reduction_factor=1.0, spectral_floor=0.1):
        """
        Remove the analyzed noise from a song
        
        Methods:
        - 'spectral_subtraction': Subtract noise spectrum from song spectrum
        - 'harmonic_subtraction': Subtract synthesized harmonic signal
        - 'adaptive_filter': Use adaptive filtering approach
        """
        print(f"Loading song: {song_path}")
        song_audio, song_sr = librosa.load(song_path, sr=None)
        song_duration = len(song_audio) / song_sr
        print(f"Song duration: {song_duration:.2f}s at {song_sr}Hz")
        
        if self.noise_frequencies is None:
            self.analyze_noise_profile(plot=False)
        
        if method == 'spectral_subtraction':
            result = self._spectral_subtraction_method(song_audio, song_sr, 
                                                     noise_reduction_factor, spectral_floor)
        elif method == 'harmonic_subtraction':
            result = self._harmonic_subtraction_method(song_audio, song_sr, 
                                                     noise_reduction_factor)
        elif method == 'adaptive_filter':
            result = self._adaptive_filter_method(song_audio, song_sr, 
                                                noise_reduction_factor)
        else:
            raise ValueError("Method must be 'spectral_subtraction', 'harmonic_subtraction', or 'adaptive_filter'")
        
        # Save the result
        sf.write(output_path, result, song_sr)
        print(f"Cleaned audio saved to: {output_path}")
        
        return result
    
    def _spectral_subtraction_method(self, song_audio, song_sr, reduction_factor, spectral_floor):
        """
        Subtract noise spectrum from song spectrum
        """
        print("Applying spectral subtraction...")
        
        # Resample noise profile to match song sample rate if needed
        if song_sr != self.sr:
            # Adjust frequency bins for different sample rate
            freq_scale = song_sr / self.sr
            noise_profile_resampled = np.interp(
                np.arange(len(self.noise_profile)) * freq_scale,
                np.arange(len(self.noise_profile)),
                self.noise_profile
            )
        else:
            noise_profile_resampled = self.noise_profile
        
        # Process in chunks to handle long audio files
        chunk_duration = 10.0  # Process 10 seconds at a time
        chunk_samples = int(chunk_duration * song_sr)
        num_chunks = int(np.ceil(len(song_audio) / chunk_samples))
        
        result = np.zeros_like(song_audio)
        
        for i in tqdm(range(num_chunks), desc="Processing chunks"):
            start_idx = i * chunk_samples
            end_idx = min(start_idx + chunk_samples, len(song_audio))
            chunk = song_audio[start_idx:end_idx]
            
            # STFT of song chunk
            song_stft = librosa.stft(chunk, n_fft=self.n_fft, 
                                   hop_length=self.hop_length, window=self.window)
            song_magnitude = np.abs(song_stft)
            song_phase = np.angle(song_stft)
            
            # Subtract noise profile
            noise_mag = noise_profile_resampled[:song_magnitude.shape[0], np.newaxis]
            cleaned_magnitude = song_magnitude - reduction_factor * noise_mag
            
            # Apply spectral floor to prevent artifacts
            cleaned_magnitude = np.maximum(cleaned_magnitude, 
                                         spectral_floor * song_magnitude)
            
            # Reconstruct audio
            cleaned_stft = cleaned_magnitude * np.exp(1j * song_phase)
            chunk_result = librosa.istft(cleaned_stft, hop_length=self.hop_length, 
                                       window=self.window)
            
            # Add to result
            result[start_idx:start_idx + len(chunk_result)] = chunk_result
        
        return result
    
    def _harmonic_subtraction_method(self, song_audio, song_sr, reduction_factor):
        """
        Subtract synthesized harmonic noise signal from song
        """
        print("Applying harmonic subtraction...")
        
        # Synthesize noise signal matching song duration and sample rate
        song_duration = len(song_audio) / song_sr
        noise_signal = self.synthesize_noise_signal(song_duration, song_sr)
        
        # Ensure same length
        min_length = min(len(song_audio), len(noise_signal))
        song_trimmed = song_audio[:min_length]
        noise_trimmed = noise_signal[:min_length]
        
        # Scale noise signal to match approximate level in song
        # Use cross-correlation to find best amplitude scaling
        correlations = []
        scales = np.linspace(0.001, 0.1, 100)
        
        print("Finding optimal noise scaling...")
        for scale in tqdm(scales, desc="Testing scales"):
            scaled_noise = noise_trimmed * scale
            # Measure how well the noise explains the song's spectral content
            residual = song_trimmed - scaled_noise
            correlation = np.corrcoef(song_trimmed, scaled_noise)[0, 1]
            correlations.append(abs(correlation) if not np.isnan(correlation) else 0)
        
        optimal_scale = scales[np.argmax(correlations)] * reduction_factor
        print(f"Optimal noise scale: {optimal_scale:.4f}")
        
        # Subtract scaled noise
        result = song_trimmed - noise_trimmed * optimal_scale
        
        return result
    
    def _adaptive_filter_method(self, song_audio, song_sr, reduction_factor):
        """
        Use adaptive filtering to remove noise
        """
        print("Applying adaptive filtering...")
        
        # Generate reference noise signal
        song_duration = len(song_audio) / song_sr
        reference_noise = self.synthesize_noise_signal(song_duration, song_sr)
        
        # Ensure same length
        min_length = min(len(song_audio), len(reference_noise))
        song_trimmed = song_audio[:min_length]
        reference_trimmed = reference_noise[:min_length]
        
        # Simple LMS adaptive filter
        filter_length = 512
        mu = 0.001  # Learning rate
        w = np.zeros(filter_length)  # Filter weights
        
        result = np.zeros_like(song_trimmed)
        
        print("Running adaptive filter...")
        for i in tqdm(range(filter_length, len(song_trimmed)), desc="Filtering"):
            # Get reference signal segment
            x = reference_trimmed[i-filter_length:i]
            
            # Filter output
            y = np.dot(w, x)
            
            # Error signal (desired - filtered reference)
            e = song_trimmed[i] - reduction_factor * y
            
            # Update filter weights
            w += mu * e * x
            
            # Store result
            result[i] = e
        
        return result
    
    def compare_methods(self, song_path, noise_reduction_factor=1.0):
        """
        Compare different noise removal methods
        """
        print("Comparing all noise removal methods...")
        
        # Method 1: Spectral Subtraction
        result1 = self.remove_noise_from_audio(song_path, 
                                             "cleaned_spectral.wav", 
                                             method='spectral_subtraction',
                                             noise_reduction_factor=noise_reduction_factor)
        
        # Method 2: Harmonic Subtraction  
        result2 = self.remove_noise_from_audio(song_path,
                                             "cleaned_harmonic.wav",
                                             method='harmonic_subtraction', 
                                             noise_reduction_factor=noise_reduction_factor)
        
        # Method 3: Adaptive Filter
        result3 = self.remove_noise_from_audio(song_path,
                                             "cleaned_adaptive.wav",
                                             method='adaptive_filter',
                                             noise_reduction_factor=noise_reduction_factor)
        
        print("\nComparison complete! Generated files:")
        print("- cleaned_spectral.wav (spectral subtraction)")
        print("- cleaned_harmonic.wav (harmonic subtraction)")
        print("- cleaned_adaptive.wav (adaptive filtering)")

# Example usage
if __name__ == "__main__":
    # Initialize with your noise sample
    remover = HarmonicBuzzRemover("zannana.mp3")
    
    # Analyze the noise profile
    remover.analyze_noise_profile(plot=True)
    
    # Remove noise from song - try different methods
    print("\n" + "="*50)
    print("Removing noise from song.mp3...")
    
    # Method 1: Spectral subtraction (recommended for steady noise)
    print("\n=== Spectral Subtraction Method ===")
    remover.remove_noise_from_audio("song.mp3", 
                                   "song_cleaned_spectral.wav",
                                   method='spectral_subtraction',
                                   noise_reduction_factor=1.2,
                                   spectral_floor=0.1)
    
    # Method 2: Harmonic subtraction (good for tonal noise)
    print("\n=== Harmonic Subtraction Method ===")
    remover.remove_noise_from_audio("song.mp3",
                                   "song_cleaned_harmonic.wav", 
                                   method='harmonic_subtraction',
                                   noise_reduction_factor=0.8)
    
    print("\n" + "="*50)
    print("Noise removal complete!")
    print("Generated files:")
    print("- song_cleaned_spectral.wav")
    print("- song_cleaned_harmonic.wav")
    print("- noise_analysis.png")
    print("\nTry both cleaned versions to see which works better.")
    print("You can adjust noise_reduction_factor (0.5-2.0) for different levels of removal.")