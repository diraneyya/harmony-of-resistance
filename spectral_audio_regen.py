import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
import threading
import sounddevice as sd
from tqdm import tqdm

class SpectralAudioRegenerator:
    def __init__(self, audio_file_path):
        """
        Initialize with your 2-second buzzing audio file
        """
        self.audio, self.sr = librosa.load(audio_file_path, sr=None)
        self.duration = len(self.audio) / self.sr
        print(f"Loaded audio: {self.duration:.2f}s at {self.sr}Hz")
        
        # STFT parameters
        self.n_fft = 2048
        self.hop_length = 512
        self.window = 'hann'
        
        # Spectral analysis results
        self.avg_magnitude = None
        self.avg_phase = None
        self.stable_spectrum = None
        
    def analyze_spectrum(self, plot=True):
        """
        Analyze the spectral content and create a stable spectral template
        """
        print("Analyzing spectral components...")
        
        # Perform STFT
        stft = librosa.stft(self.audio, 
                           n_fft=self.n_fft, 
                           hop_length=self.hop_length, 
                           window=self.window)
        
        # Get magnitude and phase
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Create stable spectral template by averaging magnitude across time
        self.avg_magnitude = np.mean(magnitude, axis=1)
        
        # For phase, we'll use a more sophisticated approach
        # Average the instantaneous frequency to get stable phase evolution
        instantaneous_freq = np.diff(np.unwrap(phase, axis=1), axis=1)
        avg_inst_freq = np.mean(instantaneous_freq, axis=1)
        
        # Create a stable phase template
        time_frames = magnitude.shape[1]
        self.avg_phase = np.zeros_like(phase)
        for i, freq in enumerate(avg_inst_freq):
            self.avg_phase[i, :] = np.linspace(0, freq * time_frames, time_frames)
        
        # Store the stable spectrum
        self.stable_spectrum = self.avg_magnitude * np.exp(1j * self.avg_phase[:, 0:1])
        
        if plot:
            self.plot_analysis(magnitude, self.avg_magnitude)
        
        print("Spectral analysis complete!")
        return self.avg_magnitude, self.avg_phase
    
    def plot_analysis(self, original_magnitude, avg_magnitude):
        """
        Plot the spectral analysis results - SAVES TO FILE, NO GUI
        """
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        
        # Set matplotlib to non-interactive backend to avoid GUI hanging
        plt.ioff()  # Turn off interactive mode
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Original spectrogram
        ax1 = axes[0, 0]
        librosa.display.specshow(librosa.amplitude_to_db(original_magnitude),
                                y_axis='hz', x_axis='time', sr=self.sr,
                                hop_length=self.hop_length, ax=ax1)
        ax1.set_title('Original Spectrogram')
        
        # Average magnitude spectrum
        ax2 = axes[0, 1]
        ax2.plot(freqs[:len(freqs)//2], avg_magnitude[:len(freqs)//2])
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude')
        ax2.set_title('Average Magnitude Spectrum')
        ax2.set_xlim(0, self.sr//2)
        
        # Dominant frequencies
        ax3 = axes[1, 0]
        # Find peaks in the spectrum
        peaks, _ = signal.find_peaks(avg_magnitude, height=np.max(avg_magnitude)*0.1)
        peak_freqs = freqs[peaks]
        peak_mags = avg_magnitude[peaks]
        
        ax3.stem(peak_freqs, peak_mags, basefmt=' ')
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Magnitude')
        ax3.set_title('Dominant Frequency Components')
        ax3.set_xlim(0, min(5000, self.sr//2))  # Focus on lower frequencies
        
        # Hide the 4th subplot
        axes[1, 1].axis('off')
        
        # Print dominant frequencies
        sorted_peaks = sorted(zip(peak_freqs, peak_mags), key=lambda x: x[1], reverse=True)
        print("\nTop 10 Dominant Frequencies:")
        for i, (freq, mag) in enumerate(sorted_peaks[:10]):
            print(f"{i+1}: {freq:.1f} Hz (magnitude: {mag:.3f})")
        
        plt.tight_layout()
        
        # Save plot instead of showing it to avoid GUI issues
        plt.savefig('spectral_analysis.png', dpi=150, bbox_inches='tight')
        print("Spectral analysis plot saved as 'spectral_analysis.png'")
        
        # Close the figure to free memory
        plt.close(fig)
        
        # Turn interactive mode back on for potential future plots
        plt.ion()
    
    def generate_infinite_audio(self, duration_seconds=60, method='spectral_repeat'):
        """
        Generate infinite audio from spectral components
        
        Methods:
        - 'spectral_repeat': Repeat the stable spectral pattern
        - 'phase_vocoder': Use phase vocoder for seamless extension
        - 'harmonic_synthesis': Synthesize from dominant harmonics
        """
        print(f"Generating {duration_seconds}s of audio using method: {method}")
        
        if self.avg_magnitude is None:
            self.analyze_spectrum(plot=False)
        
        if method == 'spectral_repeat':
            return self._spectral_repeat_method(duration_seconds)
        elif method == 'phase_vocoder':
            return self._phase_vocoder_method(duration_seconds)
        elif method == 'harmonic_synthesis':
            return self._harmonic_synthesis_method(duration_seconds)
        else:
            raise ValueError("Unknown method. Use 'spectral_repeat', 'phase_vocoder', or 'harmonic_synthesis'")
    
    def _spectral_repeat_method(self, duration_seconds):
        """
        Repeat the stable spectral pattern with slight randomization to avoid artifacts
        """
        target_samples = int(duration_seconds * self.sr)
        
        # More efficient approach: generate in larger chunks
        chunk_duration = 2.0  # Generate 2-second chunks
        chunk_samples = int(chunk_duration * self.sr)
        num_chunks = int(np.ceil(duration_seconds / chunk_duration))
        
        print(f"Generating {num_chunks} chunks of {chunk_duration}s each...")
        
        # Generate one representative chunk with the spectral characteristics
        chunk_frames = int(chunk_duration * self.sr / self.hop_length)
        
        # Create magnitude matrix for one chunk
        magnitude_matrix = np.tile(self.avg_magnitude[:, np.newaxis], (1, chunk_frames))
        
        # Create smooth phase evolution for one chunk
        phase_matrix = np.zeros((len(self.avg_magnitude), chunk_frames))
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        
        print("Creating spectral template...")
        for i in tqdm(range(len(self.avg_magnitude)), desc="Building phase matrix"):
            freq_hz = freqs[i]
            if freq_hz > 0:
                # Smooth phase evolution based on frequency
                angular_freq = 2 * np.pi * freq_hz
                time_points = np.arange(chunk_frames) * self.hop_length / self.sr
                phase_matrix[i, :] = angular_freq * time_points + np.random.uniform(0, 2*np.pi)
        
        # Create the base chunk
        print("Converting to time domain...")
        complex_stft = magnitude_matrix * np.exp(1j * phase_matrix)
        base_chunk = librosa.istft(complex_stft, 
                                  hop_length=self.hop_length, 
                                  window=self.window)
        
        # Now repeat this chunk with variations to create the full duration
        output_audio = np.zeros(target_samples)
        
        print("Assembling final audio...")
        for i in tqdm(range(num_chunks), desc="Generating chunks"):
            start_sample = i * chunk_samples
            end_sample = min(start_sample + len(base_chunk), target_samples)
            
            # Add slight variations to each repetition
            chunk_copy = base_chunk.copy()
            if i > 0:  # Add slight variations after first chunk
                # Very subtle amplitude modulation
                mod_freq = 0.1 + np.random.uniform(-0.05, 0.05)
                t = np.linspace(0, len(chunk_copy)/self.sr, len(chunk_copy))
                amplitude_mod = 1 + 0.02 * np.sin(2 * np.pi * mod_freq * t)
                chunk_copy *= amplitude_mod
            
            # Copy to output with overlap-add to avoid clicks
            if start_sample + len(chunk_copy) <= target_samples:
                output_audio[start_sample:start_sample + len(chunk_copy)] += chunk_copy
            else:
                remaining = target_samples - start_sample
                output_audio[start_sample:target_samples] += chunk_copy[:remaining]
                break
        
        print("Spectral repeat method complete!")
        return output_audio
    
    def _phase_vocoder_method(self, duration_seconds):
        """
        Use phase vocoder technique for seamless time stretching
        """
        print("Applying phase vocoder time stretching...")
        
        # Calculate stretch factor
        stretch_factor = duration_seconds / self.duration
        print(f"Stretching by factor of {stretch_factor:.1f}x")
        
        # Use librosa's phase vocoder
        stft = librosa.stft(self.audio, n_fft=self.n_fft, hop_length=self.hop_length)
        
        with tqdm(total=1, desc="Phase vocoder processing") as pbar:
            stretched_stft = librosa.phase_vocoder(stft, rate=1/stretch_factor)
            pbar.update(1)
        
        print("Converting back to time domain...")
        result = librosa.istft(stretched_stft, hop_length=self.hop_length)
        
        print("Phase vocoder method complete!")
        return result
    
    def _harmonic_synthesis_method(self, duration_seconds):
        """
        Synthesize from dominant harmonic components
        """
        print("Finding dominant frequencies...")
        
        # Find peaks in the spectrum
        peaks, properties = signal.find_peaks(self.avg_magnitude, 
                                            height=np.max(self.avg_magnitude)*0.05,
                                            distance=10)
        
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        peak_freqs = freqs[peaks]
        peak_amplitudes = self.avg_magnitude[peaks]
        
        print(f"Found {len(peak_freqs)} dominant frequencies")
        
        # Generate time vector
        t = np.linspace(0, duration_seconds, int(duration_seconds * self.sr))
        
        # Synthesize audio from harmonics
        synthesized = np.zeros_like(t)
        
        print("Synthesizing from harmonics...")
        for i, (freq, amp) in enumerate(tqdm(zip(peak_freqs, peak_amplitudes), 
                                           desc="Adding harmonics", 
                                           total=len(peak_freqs))):
            if freq > 0:  # Skip DC component
                # Pure steady harmonics - no modulation for perfect stability
                synthesized += amp * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2*np.pi))
        
        # Normalize
        synthesized = synthesized / (np.max(np.abs(synthesized)) + 1e-8)
        
        print("Harmonic synthesis complete!")
        return synthesized
    
    def save_audio(self, audio, filename, normalize=True):
        """
        Save generated audio to file
        """
        if normalize:
            audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.9
        
        sf.write(filename, audio, self.sr)
        print(f"Audio saved to: {filename}")
    
    def play_infinite_loop(self, chunk_duration=10, method='spectral_repeat'):
        """
        Play audio in an infinite loop with real-time generation
        """
        print("Starting infinite playback... Press Ctrl+C to stop")
        
        try:
            while True:
                # Generate a chunk
                audio_chunk = self.generate_infinite_audio(chunk_duration, method)
                
                # Play the chunk
                sd.play(audio_chunk, self.sr)
                sd.wait()  # Wait for playback to finish before generating next chunk
                
        except KeyboardInterrupt:
            print("\nPlayback stopped by user")
            sd.stop()

# Example usage
if __name__ == "__main__":
    # Initialize with your audio file
    regenerator = SpectralAudioRegenerator("zannana.mp3")
    
    # Analyze the spectral content - saves plot to file instead of showing GUI
    regenerator.analyze_spectrum(plot=True)
    
    # Generate different versions
    print("\nGenerating audio with different methods...")
    
    # Method 1: Spectral repeat (recommended for buzzing sounds)
    print("\n=== Method 1: Spectral Repeat ===")
    audio1 = regenerator.generate_infinite_audio(30, method='spectral_repeat')
    regenerator.save_audio(audio1, "zannana_infinite_spectral.wav")
    
    # Method 2: Harmonic synthesis
    print("\n=== Method 2: Harmonic Synthesis ===")
    audio2 = regenerator.generate_infinite_audio(30, method='harmonic_synthesis')
    regenerator.save_audio(audio2, "zannana_infinite_harmonic.wav")
    
    # Method 3: Phase vocoder (time stretching)
    print("\n=== Method 3: Phase Vocoder ===")
    audio3 = regenerator.generate_infinite_audio(30, method='phase_vocoder')
    regenerator.save_audio(audio3, "zannana_infinite_stretched.wav")
    
    print("\n" + "="*50)
    print("All methods complete!")
    print("Generated files:")
    print("- zannana_infinite_spectral.wav")
    print("- zannana_infinite_harmonic.wav") 
    print("- zannana_infinite_stretched.wav")
    print("- spectral_analysis.png")
    print("\nTry each generated file to see which works best.")
    print("\nTo play infinite loop in real-time, use:")
    print("regenerator.play_infinite_loop(chunk_duration=10, method='spectral_repeat')")
    
    # Uncomment the line below to start infinite playback immediately
    # regenerator.play_infinite_loop(chunk_duration=10, method='spectral_repeat')