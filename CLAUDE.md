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
- **Drone Audio**: Infinite harmonic reconstruction (`zannana_improved_exact.wav`)
- **Loading System**: Preloads both video and audio files with animated loading screen

## File Structure
```
project/
├── index.html                          # Main web application
├── song_cleaned.mp4                    # Video without drone audio
├── zannana_improved_exact.wav          # Infinite drone audio loop
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
- **Smooth Transitions**: 50ms fade in/out when toggling
- **Volume Range**: Drone audio limited to 20% maximum volume for optimal mixing

### User Interface
- **Loading Animation**: Animated spinner with progress bar during file preloading
- **Drone Volume Control**: 
  - Desktop: Smooth slider for precise volume adjustment (0-100%)
  - Mobile: Toggle switch between 0% and 50% (optimized for touch devices)
- **Status Display**: Dynamic text with gradient color system (yellow → orange based on volume percentage)
- **Responsive Design**: Works on mobile and desktop with adaptive layouts
- **Keyboard Controls**: 
  - Spacebar: Play/pause video
  - 'D' key: Toggle drone volume (0% ↔ 50%)

### Technical Challenges Solved
1. **Media Preloading**: Loading animation ensures both video and audio files are cached before interaction
2. **No Page Reload**: Uses separate audio element + Web Audio API instead of video source switching
3. **Audio Sync**: Periodic sync checks prevent drift between video and drone audio
4. **Browser Compatibility**: Graceful fallback for browsers without Web Audio API support
5. **Mobile Support**: Touch event handling and audio context resumption
6. **Adaptive Controls**: Automatic detection switches between slider (desktop) and toggle (mobile)
7. **Loading States**: Error handling and fallback timeout for failed media loads

## Deployment (GitHub Pages)
1. Create new repository
2. Upload files: `index.html`, `song_cleaned.mp4`, `zannana_improved_exact.wav`
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
- **Media Preloading**: Loading screen ensures files are cached before user interaction
- **Progressive Loading**: Visual feedback with progress bar during file loading
- Use `createMediaElementSource()` to avoid audio copying
- Minimal DOM manipulation during playback
- Efficient sync checking (1-second intervals)
- **Loading Timeout**: 10-second fallback prevents infinite loading states

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

## Recent Updates

### Mobile Toggle Implementation (Latest)
- **Adaptive Interface**: CSS media queries hide slider on mobile (≤768px) and show toggle
- **Binary Control**: Simple on/off toggle switching between 0% and 50% drone volume
- **Touch Optimization**: Large 80×40px toggle with smooth animations for finger interaction
- **Visual Feedback**: Orange gradient background and sliding knob animation when active
- **Label System**: Dynamic "Off" and "Drone 50%" labels with active state highlighting
- **JavaScript Integration**: `handleMobileToggle()` function maintains sync with slider logic
- **Consistent Behavior**: Mobile toggle triggers same audio processing as desktop slider

### Gradient Color System
- **Dynamic Status Colors**: Banner color transitions smoothly from yellow (0%) to orange (100%)
- **Real-time Updates**: Color changes instantly as user adjusts volume slider/toggle
- **RGB Calculation**: Yellow (255,255,0) interpolates to Orange (255,165,0) based on volume ratio
- **CSS Integration**: Inline styles override default classes for gradient effect
- **Accessibility**: Maintains sufficient contrast ratios across the color spectrum

### Loading System Implementation
- **Full-Screen Loading Overlay**: Covers page content during media loading
- **Animated Progress Tracking**: Visual spinner and progress bar (0-100%)
- **Dual Media Loading**: Monitors both video (`loadeddata`) and audio (`canplaythrough`) events
- **Error Resilience**: Continues loading even if media files fail to load
- **Smooth Transitions**: 500ms fade-out animation when loading completes
- **Mobile Responsive**: Adaptive spinner and text sizing for smaller screens
- **Loading States**: Dynamic text updates ("Preparing files..." → "Ready to experience...")

### CSS Enhancements
- Consistent theme colors (#ff4444 red accent matching drone UI)
- Backdrop blur effects for modern glass-morphism aesthetic
- Responsive breakpoints for mobile/desktop optimization
- Loading animation keyframes with smooth 1s rotation cycle

### JavaScript Architecture
- Event-driven loading system with `setupMediaLoading()` function
- Progress state tracking with dual boolean flags (video/audio)
- Automatic DOM cleanup after loading completes
- Fallback timeout system (10 seconds) for network issues

## Technical Debt
- Consider implementing audio worklets for better performance
- Add service worker for offline functionality
- Optimize audio file compression for faster loading
- ✅ **RESOLVED**: Add error handling for failed audio loads (now implemented with loading system)