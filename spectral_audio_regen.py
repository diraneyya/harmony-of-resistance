import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
import threading
import sounddevice as sd

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
        Plot the spectral analysis results
        """
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        
        plt.figure(figsize=(15, 10))
        
        # Original spectrogram
        plt.subplot(2, 2, 1)
        librosa.display.specshow(librosa.amplitude_to_db(original_magnitude),
                                y_axis='hz', x_axis='time', sr=self.sr,
                                hop_length=self.hop_length)
        plt.title('Original Spectrogram')
        plt.colorbar(format='%+2.0f dB')
        
        # Average magnitude spectrum
        plt.subplot(2, 2, 2)
        plt.plot(freqs[:len(freqs)//2], avg_magnitude[:len(freqs)//2])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title('Average Magnitude Spectrum')
        plt.xlim(0, self.sr//2)
        
        # Dominant frequencies
        plt.subplot(2, 2, 3)
        # Find peaks in the spectrum
        peaks, _ = signal.find_peaks(avg_magnitude, height=np.max(avg_magnitude)*0.1)
        peak_freqs = freqs[peaks]
        peak_mags = avg_magnitude[peaks]
        
        plt.stem(peak_freqs, peak_mags, basefmt=' ')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title('Dominant Frequency Components')
        plt.xlim(0, min(5000, self.sr//2))  # Focus on lower frequencies
        
        # Print dominant frequencies
        sorted_peaks = sorted(zip(peak_freqs, peak_mags), key=lambda x: x[1], reverse=True)
        print("\nTop 10 Dominant Frequencies:")
        for i, (freq, mag) in enumerate(sorted_peaks[:10]):
            print(f"{i+1}: {freq:.1f} Hz (magnitude: {mag:.3f})")
        
        plt.tight_layout()
        plt.show()
    
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
        output_audio = np.zeros(target_samples)
        
        # Parameters for generation
        chunk_length = self.hop_length * (self.avg_magnitude.shape[0] // self.hop_length)
        
        # Generate audio in chunks
        current_pos = 0
        while current_pos < target_samples:
            # Create STFT with stable magnitude and evolving phase
            chunk_frames = min(100, (target_samples - current_pos) // self.hop_length + 1)
            
            # Create magnitude matrix
            magnitude_matrix = np.tile(self.avg_magnitude[:, np.newaxis], (1, chunk_frames))
            
            # Create phase matrix with continuous evolution
            phase_matrix = np.zeros((len(self.avg_magnitude), chunk_frames))
            for i in range(len(self.avg_magnitude)):
                # Create smooth phase evolution
                freq_bin = i
                angular_freq = 2 * np.pi * freq_bin * self.hop_length / self.n_fft
                phase_matrix[i, :] = angular_freq * np.arange(chunk_frames) + np.random.uniform(0, 2*np.pi)
            
            # Combine magnitude and phase
            complex_stft = magnitude_matrix * np.exp(1j * phase_matrix)
            
            # Convert back to time domain
            chunk_audio = librosa.istft(complex_stft, 
                                       hop_length=self.hop_length, 
                                       window=self.window)
            
            # Add to output
            end_pos = min(current_pos + len(chunk_audio), target_samples)
            output_audio[current_pos:end_pos] = chunk_audio[:end_pos-current_pos]
            current_pos = end_pos
        
        return output_audio
    
    def _phase_vocoder_method(self, duration_seconds):
        """
        Use phase vocoder technique for seamless time stretching
        """
        # Calculate stretch factor
        stretch_factor = duration_seconds / self.duration
        
        # Use librosa's phase vocoder
        stft = librosa.stft(self.audio, n_fft=self.n_fft, hop_length=self.hop_length)
        stretched_stft = librosa.phase_vocoder(stft, rate=1/stretch_factor)
        
        return librosa.istft(stretched_stft, hop_length=self.hop_length)
    
    def _harmonic_synthesis_method(self, duration_seconds):
        """
        Synthesize from dominant harmonic components
        """
        # Find peaks in the spectrum
        peaks, properties = signal.find_peaks(self.avg_magnitude, 
                                            height=np.max(self.avg_magnitude)*0.05,
                                            distance=10)
        
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        peak_freqs = freqs[peaks]
        peak_amplitudes = self.avg_magnitude[peaks]
        
        # Generate time vector
        t = np.linspace(0, duration_seconds, int(duration_seconds * self.sr))
        
        # Synthesize audio from harmonics
        synthesized = np.zeros_like(t)
        
        for freq, amp in zip(peak_freqs, peak_amplitudes):
            if freq > 0:  # Skip DC component
                # Add slight frequency and amplitude modulation for more natural sound
                freq_mod = freq * (1 + 0.001 * np.sin(2 * np.pi * 0.5 * t))  # Slight vibrato
                amp_mod = amp * (1 + 0.05 * np.sin(2 * np.pi * 0.3 * t))     # Slight tremolo
                
                synthesized += amp_mod * np.sin(2 * np.pi * freq_mod * t + np.random.uniform(0, 2*np.pi))
        
        # Normalize
        synthesized = synthesized / (np.max(np.abs(synthesized)) + 1e-8)
        
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
    
    # Analyze the spectral content
    regenerator.analyze_spectrum(plot=True)
    
    # Generate different versions
    print("\nGenerating audio with different methods...")
    
    # Method 1: Spectral repeat (recommended for buzzing sounds)
    audio1 = regenerator.generate_infinite_audio(30, method='spectral_repeat')
    regenerator.save_audio(audio1, "zannana_infinite_spectral.wav")
    
    # Method 2: Harmonic synthesis
    audio2 = regenerator.generate_infinite_audio(30, method='harmonic_synthesis')
    regenerator.save_audio(audio2, "zannana_infinite_harmonic.wav")
    
    # Method 3: Phase vocoder (time stretching)
    audio3 = regenerator.generate_infinite_audio(30, method='phase_vocoder')
    regenerator.save_audio(audio3, "zannana_infinite_stretched.wav")
    
    print("\nAll methods complete! Try each generated file to see which works best.")
    print("\nTo play infinite loop in real-time, use:")
    print("regenerator.play_infinite_loop(chunk_duration=10, method='spectral_repeat')")
    
    # Uncomment the line below to start infinite playback immediately
    # regenerator.play_infinite_loop(chunk_duration=10, method='spectral_repeat')