# Drone Resistance Music Player - Development Notes

## Project Overview
A web application that demonstrates artistic resistance through music - showcasing how a musician living under constant drone surveillance transformed the oppressive sound into an instrument of creative defiance.

## Technical Implementation

### Audio Processing Pipeline
1. **Original Audio Sample**: `zannana.mp3` (0.59s buzz/drone recording)
2. **Spectral Analysis**: Extracted harmonic components using librosa STFT
3. **Infinite Generation**: Created seamless looping drone audio using harmonic synthesis
4. **Noise Removal**: Applied advanced spectral subtraction to create cleaned version

### Key Audio Frequencies Identified
- 344.5 Hz (primary, amplitude: 42.791)
- 581.4 Hz (amplitude: 26.136) 
- 925.9 Hz (amplitude: 22.567)
- 107.7 Hz (amplitude: 8.509)
- 1399.7 Hz (amplitude: 4.709)
- 1162.8 Hz (amplitude: 3.390)

### Web Application Architecture
- **Frontend**: Single HTML file with embedded CSS/JS
- **Audio Mixing**: Web Audio API for real-time drone layering
- **Video Source**: Cleaned version only (`song_cleaned.mp4`)
- **Drone Audio**: Infinite harmonic reconstruction (`zannana_infinite_harmonic.wav`)

## File Structure
```
project/
├── index.html                          # Main web application
├── song_cleaned.mp4                    # Video without drone audio
├── zannana_infinite_harmonic.wav       # Infinite drone audio loop
├── spectral_audio_regen.py            # Original audio generation script
├── harmonic_buzz_removal.py           # Noise removal tool
├── improved_buzz_remover.py           # Advanced noise removal
└── CLAUDE.md                          # This file
```

## Python Dependencies
```bash
pip install numpy librosa soundfile matplotlib scipy tqdm
```

## Audio Processing Scripts

### 1. Spectral Audio Regenerator (`spectral_audio_regen.py`)
- Analyzes buzz sample using STFT
- Generates infinite audio using three methods:
  - Spectral repeat (recommended for steady noise)
  - Harmonic synthesis (pure sine wave reconstruction)
  - Phase vocoder (time stretching)

### 2. Harmonic Buzz Remover (`harmonic_buzz_removal.py`)
- Advanced spectral subtraction with overlapping chunks
- Multiple noise removal methods
- Adjustable parameters for fine-tuning

Key parameters for spectral subtraction:
- `noise_reduction_factor`: 1.0-3.0 (aggressiveness of removal)
- `spectral_floor`: 0.01-0.2 (preservation of original signal)

## Web Application Features

### Audio Mixing Implementation
- **Web Audio API**: Real-time audio graph mixing
- **Fallback Mode**: Simple volume control for unsupported browsers
- **Synchronization**: Keeps drone audio locked to video timeline
- **Smooth Transitions**: 100ms fade in/out when toggling

### User Interface
- **Drone Toggle**: Visual switch with custom SVG icon
- **Status Display**: Dynamic text showing current playback mode
- **Responsive Design**: Works on mobile and desktop
- **Keyboard Controls**: 
  - Spacebar: Play/pause video
  - 'D' key: Toggle drone audio

### Technical Challenges Solved
1. **No Page Reload**: Uses separate audio element + Web Audio API instead of video source switching
2. **Audio Sync**: Periodic sync checks prevent drift between video and drone audio
3. **Browser Compatibility**: Graceful fallback for browsers without Web Audio API support
4. **Mobile Support**: Touch event handling and audio context resumption

## Deployment (GitHub Pages)
1. Create new repository
2. Upload files: `index.html`, `song_cleaned.mp4`, `zannana_infinite_harmonic.wav`
3. Enable Pages in repository settings
4. Site available at: `https://username.github.io/repository-name`

## FFmpeg Commands Used

### Audio Replacement in Video
```bash
# Replace audio in MP4 with cleaned version
ffmpeg -i song.mp4 -i song_cleaned_spectral.wav -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 song_cleaned.mp4
```

## Development Notes

### Audio Analysis Results
The spectral analysis revealed the drone consists primarily of harmonic frequencies, making it suitable for:
- Harmonic synthesis for infinite generation
- Targeted spectral subtraction for removal
- Real-time mixing without artifacts

### Performance Optimizations
- Preload drone audio with `preload="auto"`
- Use `createMediaElementSource()` to avoid audio copying
- Minimal DOM manipulation during playback
- Efficient sync checking (1-second intervals)

### Browser Considerations
- **Chrome/Safari**: Full Web Audio API support
- **Firefox**: Good support with minor differences
- **Mobile Safari**: Requires user interaction before audio context creation
- **Fallback**: Simple volume control when Web Audio unavailable

## Artistic Context
The project demonstrates how oppressive surveillance technology can be transformed into artistic expression. The ability to toggle between "with drone" and "without drone" versions highlights how the musician incorporated the invasive sound into their creative process, turning a symbol of fear into a foundation for resistance.

## Future Enhancements
- Visual waveform display during drone playback
- Multiple drone audio layers for different intensity levels
- Real-time audio visualization
- Social sharing functionality
- Multi-language support for broader impact

## Technical Debt
- Consider implementing audio worklets for better performance
- Add service worker for offline functionality
- Optimize audio file compression for faster loading
- Add error handling for failed audio loads