# Drone Resistance Music Player - Development Notes

## Project Overview
A web application that demonstrates artistic resistance through music - showcasing how a musician living under constant drone surveillance transformed the oppressive sound into an instrument of creative defiance.

## Technical Implementation

### Audio Processing Pipeline
1. **Original Audio Sample**: `zannana.mp3` (0.59s buzz/drone recording)
2. **Spectral Analysis**: Extracted harmonic components using librosa STFT
3. **Infinite Generation**: Created seamless looping drone audio using harmonic synthesis
4. **Noise Removal**: Applied spectral subtraction to create cleaned mono version

### Key Audio Frequencies Identified (High-Resolution Analysis)
- 344.5 Hz (primary, amplitude: 42.791)
- 581.4 Hz (amplitude: 26.136) 
- 925.9 Hz (amplitude: 22.567)
- 107.7 Hz (amplitude: 8.509)
- 1399.7 Hz (amplitude: 4.709)
- 1162.8 Hz (amplitude: 3.390)
- **50+ additional harmonic components detected with improved analysis**

### Web Application Architecture
- **Frontend**: Single HTML file with embedded CSS/JS
- **Audio Mixing**: Web Audio API for real-time drone layering
- **Video Source**: Cleaned mono version only (`song_cleaned.mp4`)
- **Drone Audio**: Perfect harmonic reconstruction (`zannana_improved_exact.wav`)

## File Structure
```
project/
├── index.html                          # Main web application
├── song_cleaned.mp4                    # Video without drone audio (mono)
├── zannana_improved_exact.wav          # Perfect infinite drone reconstruction
├── spectral_audio_regen.py            # Original multi-method generator
├── reconstruct_zannana.py              # High-resolution drone generator (improved)
├── harmonic_buzz_removal.py           # Original noise removal tool
├── harmonic_buzz_removal2.py          # Stereo-aware noise removal (failed)
├── stereo_restore.py                  # Stereo restoration attempts (failed)
└── CLAUDE.md                          # This file
```

## Python Dependencies
```bash
pip install numpy librosa soundfile matplotlib scipy tqdm
```

## Audio Processing Scripts

### 1. Spectral Audio Regenerator (`spectral_audio_regen.py`)
Original multi-method generator with three approaches:
- Spectral repeat (recommended for steady noise)
- Harmonic synthesis (pure sine wave reconstruction) ⭐ **This became the foundation**
- Phase vocoder (time stretching)

### 2. Reconstruct Zannana (`reconstruct_zannana.py`)
**Key breakthrough script** - High-resolution harmonic synthesizer:
- **8192-point FFT** (4x higher resolution than original)
- **256 hop length** for maximum precision
- **Lower threshold detection** (0.01 vs 0.05) captures subtle components
- **All components included**: Uses 50+ frequency components vs original 6
- **Perfect phase relationships**: Random but consistent phase initialization
- **Automatic video duration matching**: Uses ffprobe to get exact video length

Critical parameters that produced "incredible" results:
```python
self.n_fft = 8192        # Much higher resolution
self.hop_length = 256    # Smaller hop for precision
height=np.max(avg_magnitude)*0.01  # Lower threshold
distance=5               # Allows closer peaks
```

### 3. Harmonic Buzz Remover (`harmonic_buzz_removal.py`)
**Working mono noise removal tool**:
- Spectral subtraction with optimal parameters
- reduction_factor=1.0, spectral_floor=0.01
- Chunk-based processing (10-second chunks)
- Produces high-quality drone-free mono audio

### 4. Harmonic Buzz Remover 2 (`harmonic_buzz_removal2.py`)
**Failed stereo attempt** - Channel-wise processing:
- Processes each stereo channel independently
- Uses same algorithm as working mono version
- **Result**: Much worse quality, drone sound very clear in output
- **Conclusion**: Spectral subtraction may only work effectively on mono

### 5. Stereo Restore (`stereo_restore.py`)
**Failed stereo restoration attempts**:
- Frequency-selective restoration (replace only drone frequencies)
- Side information restoration (mid/side processing)
- **Result**: Drone sound came back very strongly in both methods
- **Conclusion**: Dead end for stereo processing

## Web Application Features

### Audio Mixing Implementation
- **Web Audio API**: Real-time audio graph mixing
- **Perfect Synchronization**: Drone audio locked to video timeline with 1-second sync checks
- **Smooth Transitions**: 100ms fade in/out when toggling
- **Fallback Mode**: Simple volume control for unsupported browsers

### User Interface
- **Drone Toggle**: Visual switch with custom SVG icon
- **Status Display**: Dynamic text showing current playback mode
- **Responsive Design**: Works on mobile and desktop
- **Keyboard Controls**: 
  - Spacebar: Play/pause video
  - 'D' key: Toggle drone audio

## Deployment (GitHub Pages)
1. Create new repository
2. Upload files: `index.html`, `song_cleaned.mp4`, `zannana_improved_exact.wav`
3. Enable Pages in repository settings
4. Site available at: `https://username.github.io/repository-name`

## FFmpeg Commands Used

### Get Video Duration
```bash
# Get duration in seconds for Python scripts
ffprobe -v quiet -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 song_cleaned.mp4

# Human readable
ffmpeg -i song_cleaned.mp4 2>&1 | grep Duration
```

### Audio Replacement in Video (Mono)
```bash
# Replace audio in MP4 with mono cleaned version
ffmpeg -i song.mp4 -i song_cleaned_spectral.wav -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 song_cleaned.mp4
```

## Development Notes

### Audio Analysis Breakthrough
**Most Important Discovery**: The improved harmonic synthesizer (`reconstruct_zannana.py`) with 8192-point FFT produced "incredible" results that were described as "absolutely perfect reconstruction." This became the foundation for all subsequent processing.

**Key factors for success:**
- High-resolution frequency analysis (8192 vs 2048 FFT)
- Inclusion of ALL detected frequency components (50+ vs 6)
- Lower detection threshold capturing subtle harmonics
- Pure sine wave synthesis without modulation

### Performance Optimizations
- Use `createMediaElementSource()` to avoid audio copying
- Minimal DOM manipulation during playback
- Efficient sync checking (1-second intervals)
- Preload drone audio with `preload="auto"`

### Browser Considerations
- **Chrome/Safari**: Full Web Audio API support
- **Firefox**: Good support with minor differences
- **Mobile Safari**: Requires user interaction before audio context creation
- **Fallback**: Simple volume control when Web Audio unavailable

## Lessons Learned

### What Worked
1. **Spectral method superiority**: Spectral subtraction consistently outperformed harmonic subtraction
2. **High-resolution analysis**: 8192-point FFT was crucial for perfect reconstruction
3. **Mono processing**: Mono audio processing achieved high-quality results
4. **Simple is better**: Complex modulation and advanced techniques degraded quality
5. **Optimal parameters**: reduction_factor=1.0, spectral_floor=0.01 found through experimentation

### What Failed
1. **Complex high-fidelity analyzer**: Sophisticated analysis with vibrato/tremolo detection produced poor results
2. **Harmonic subtraction**: Direct subtraction "eroded the harmonics of the song itself"
3. **Advanced modulation**: Adding realistic vibrato/tremolo moved away from desired steady drone
4. **Over-engineering**: More sophisticated != better results
5. **Stereo processing**: All stereo approaches failed significantly

### Critical Technical Insights
1. **Perfect reconstruction quality**: The improved synthesizer achieved "incredible" results that were qualitatively different from previous attempts
2. **Mono limitation discovered**: Spectral subtraction techniques appear to only work effectively on mono audio
3. **Stereo failure pattern**: Multiple stereo approaches (channel-wise processing, frequency-selective restoration, side information) all failed
4. **Parameter sensitivity**: Small changes in spectral floor (0.01 vs 0.1) made significant difference
5. **Algorithm stability**: Stick with working methods, only change format/resolution

### The Stereo Problem
**Fundamental limitation discovered**: Spectral subtraction assumes noise consistency across channels, but:
- Drone may be positioned differently in stereo field
- Phase differences exist between left/right drone content
- Musical content distribution across stereo field creates conflicts
- Stereo acoustic interactions are too complex for simple spectral subtraction

**Practical solution**: Accept mono cleaned version for web application. Many listeners won't notice the difference, especially on common playback devices.

## Artistic Context
The project demonstrates how oppressive surveillance technology can be transformed into artistic expression. The toggle functionality allows experiencing the musician's creative process - incorporating invasive drone surveillance sound into musical resistance.

## Future Enhancements
- Visual waveform display during drone playback
- Real-time audio visualization
- Social sharing functionality
- Multi-language support for broader impact
- Service worker for offline functionality

## Technical Debt
- Consider implementing audio worklets for better performance
- Optimize audio file compression for faster loading
- Add comprehensive error handling for network failures
- **Stereo processing**: Requires fundamentally different approach beyond spectral subtraction