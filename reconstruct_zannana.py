import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm
import subprocess

class ImprovedHarmonicSynthesizer:
    def __init__(self, audio_file_path):
        """
        Improved version of the original harmonic synthesizer with higher resolution
        """
        self.audio, self.sr = librosa.load(audio_file_path, sr=None)
        self.duration = len(self.audio) / self.sr
        print(f"Loaded audio: {self.duration:.2f}s at {self.sr}Hz")
        
        # Higher resolution STFT parameters
        self.n_fft = 8192        # Much higher resolution (was 2048)
        self.hop_length = 256    # Smaller hop for more precision (was 512)
        self.window = 'hann'
        
        # Analysis results
        self.noise_frequencies = None
        self.noise_amplitudes = None
        self.noise_profile = None
        
    def analyze_noise_profile(self, plot=True):
        """
        High-resolution analysis of harmonic components - same method but higher resolution
        """
        print("Analyzing noise harmonic profile with high resolution...")
        
        # Perform high-resolution STFT
        stft = librosa.stft(self.audio, 
                           n_fft=self.n_fft, 
                           hop_length=self.hop_length, 
                           window=self.window)
        
        # Get average magnitude spectrum
        magnitude = np.abs(stft)
        avg_magnitude = np.mean(magnitude, axis=1)
        
        # Find peaks with higher precision and lower threshold
        peaks, properties = signal.find_peaks(avg_magnitude, 
                                            height=np.max(avg_magnitude)*0.01,  # Lower threshold
                                            distance=5)  # Closer peaks allowed
        
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        self.noise_frequencies = freqs[peaks]
        self.noise_amplitudes = avg_magnitude[peaks]
        
        # Store the complete noise profile
        self.noise_profile = avg_magnitude
        
        print(f"Found {len(self.noise_frequencies)} frequency components (higher resolution)")
        
        if plot:
            self.plot_noise_analysis(freqs, avg_magnitude)
        
        # Print dominant frequencies with more precision
        sorted_peaks = sorted(zip(self.noise_frequencies, self.noise_amplitudes), 
                            key=lambda x: x[1], reverse=True)
        print(f"\nTop {min(20, len(sorted_peaks))} Frequency Components:")
        for i, (freq, amp) in enumerate(sorted_peaks[:20]):
            print(f"{i+1:2d}: {freq:8.2f} Hz (amplitude: {amp:.4f})")
        
        return self.noise_frequencies, self.noise_amplitudes
    
    def plot_noise_analysis(self, freqs, avg_magnitude):
        """
        Plot the high-resolution analysis results
        """
        plt.ioff()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Full spectrum with higher detail
        ax1.plot(freqs[:len(freqs)//4], avg_magnitude[:len(freqs)//4])  # Show up to fs/4
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Magnitude')
        ax1.set_title('High-Resolution Noise Spectrum')
        ax1.set_xlim(0, self.sr//4)
        ax1.grid(True, alpha=0.3)
        
        # Dominant frequencies with more detail
        peaks, _ = signal.find_peaks(avg_magnitude, 
                                   height=np.max(avg_magnitude)*0.01, 
                                   distance=5)
        peak_freqs = freqs[peaks]
        peak_mags = avg_magnitude[peaks]
        
        # Show more components
        top_indices = np.argsort(peak_mags)[-50:]  # Top 50 components
        ax2.stem(peak_freqs[top_indices], peak_mags[top_indices], basefmt=' ')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude')
        ax2.set_title('Top 50 Frequency Components')
        ax2.set_xlim(0, min(3000, self.sr//4))
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('improved_noise_analysis.png', dpi=150, bbox_inches='tight')
        print("Analysis plot saved as 'improved_noise_analysis.png'")
        plt.close(fig)
        plt.ion()
    
    def synthesize_improved_harmonic_drone(self, duration_seconds, target_sr=None):
        """
        Improved harmonic synthesis - same core method but with all detected components
        """
        if self.noise_frequencies is None:
            self.analyze_noise_profile(plot=False)
        
        if target_sr is None:
            target_sr = self.sr
        
        print(f"Synthesizing improved harmonic drone: {duration_seconds:.1f}s at {target_sr}Hz...")
        print(f"Using {len(self.noise_frequencies)} frequency components...")
        
        # Generate time vector
        t = np.linspace(0, duration_seconds, int(duration_seconds * target_sr))
        
        # Synthesize from ALL detected harmonics (not just the top ones)
        synthesized = np.zeros_like(t)
        
        # Use all components but weight them by amplitude
        for freq, amp in tqdm(zip(self.noise_frequencies, self.noise_amplitudes), 
                             desc="Synthesizing harmonics", 
                             total=len(self.noise_frequencies)):
            if freq > 0 and freq < target_sr/2:  # Valid frequency range
                # Pure steady harmonics - exactly like before but with ALL components
                phase = np.random.uniform(0, 2*np.pi)  # Random phase for each component
                component = amp * np.sin(2 * np.pi * freq * t + phase)
                synthesized += component
        
        # Normalize to prevent clipping but maintain relative levels
        if np.max(np.abs(synthesized)) > 0:
            synthesized = synthesized / np.max(np.abs(synthesized)) * 0.9
        
        print("Improved harmonic synthesis complete!")
        return synthesized
    
    def get_video_duration(self, video_path):
        """
        Get video duration using ffprobe
        """
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', 
                '-of', 'default=noprint_wrappers=1:nokey=1', video_path
            ], capture_output=True, text=True)
            
            duration = float(result.stdout.strip())
            print(f"Video duration: {duration:.2f} seconds")
            return duration
        except:
            print("Could not get video duration automatically. Using default 91.5 seconds.")
            return 91.5

# Example usage
if __name__ == "__main__":
    # Initialize improved synthesizer
    synthesizer = ImprovedHarmonicSynthesizer("zannana.mp3")
    
    # Perform high-resolution analysis
    synthesizer.analyze_noise_profile(plot=True)
    
    # Get video duration
    video_duration = synthesizer.get_video_duration("song_cleaned.mp4")
    
    # Generate improved harmonic versions
    print("\nGenerating improved harmonic drone audio...")
    
    # Short test version
    test_audio = synthesizer.synthesize_improved_harmonic_drone(10)
    sf.write("zannana_improved_test.wav", test_audio, synthesizer.sr)
    print("Saved: zannana_improved_test.wav (10s test)")
    
    # Exact video length version
    exact_audio = synthesizer.synthesize_improved_harmonic_drone(video_duration)
    sf.write("zannana_improved_exact.wav", exact_audio, synthesizer.sr)
    print(f"Saved: zannana_improved_exact.wav ({video_duration:.1f}s - exact video length)")
    
    # Extended version for looping
    extended_audio = synthesizer.synthesize_improved_harmonic_drone(video_duration * 2)
    sf.write("zannana_improved_extended.wav", extended_audio, synthesizer.sr)
    print(f"Saved: zannana_improved_extended.wav ({video_duration*2:.1f}s - extended)")
    
    print("\nImproved harmonic synthesis complete!")
    print("Files generated:")
    print("- zannana_improved_test.wav (quick test)")
    print("- zannana_improved_exact.wav (exact video length - USE THIS)")
    print("- zannana_improved_extended.wav (extended version)")
    print("- improved_noise_analysis.png (high-resolution analysis)")
    print(f"\nUpdate your web app to use: zannana_improved_exact.wav")