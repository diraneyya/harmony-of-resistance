# Resistance Through Harmony

A web-based music player that tells the story of artistic resistance under surveillance.

## About This Project

In Gaza, the constant presence of surveillance drones creates an inescapable buzzing sound that permeates daily life. Instead of being silenced by this oppressive noise, musician Ahmed Muin from *Gaza Birds Singing* discovered something remarkable: by singing in harmony with the drone's frequency, he transformed the sound of surveillance into an instrument of creative defiance.

This project brings that story to life through an interactive experience where you can toggle the drone sound on and off, understanding how the artist incorporated it into his music.

## What's Inside

This repository contains:

- **Interactive web player** (`index.html`) - A single-file application that plays the music with optional drone audio overlay
- **Audio processing tools** (Python scripts) - Scripts used to extract, analyze, and reconstruct the drone frequencies from the original recording
- **Multiple video qualities** - Adaptive bandwidth detection serves appropriate video size based on connection speed
- **Development documentation** (`CLAUDE.md`) - Technical details about the audio processing pipeline and implementation decisions

## The Technical Story

The drone sound heard in the original recording was:
1. Analyzed using spectral analysis to identify its harmonic frequencies (344.5 Hz primary, plus 50+ additional components)
2. Reconstructed as a perfect looping audio track using high-resolution FFT analysis
3. Removed from the video using spectral subtraction techniques
4. Made available as a separate audio layer that can be mixed in real-time

This allows listeners to experience the music with and without the drone sound, understanding the artist's creative process of working *with* rather than *against* the oppressive environment.

## Live Demo

Experience the interactive player at either of these locations:

- **Cloudflare Pages**: [https://harmony-of-resistance.pages.dev/](https://harmony-of-resistance.pages.dev/)
- **GitHub Pages**: [https://diraneyya.github.io/harmony-of-resistance/](https://diraneyya.github.io/harmony-of-resistance/)

## Context & Attribution

- **Artist**: Ahmed Muin, *Gaza Birds Singing*
- **Documentary**: [Al Jazeera - The Drones & The Zawaana](https://www.youtube.com/watch?v=MtodtEKmYSE)
- **Wikipedia**: [Zanana](https://en.wikipedia.org/wiki/Zanana) - Traditional Palestinian vocal art form

## License

See [LICENSE](LICENSE) file for details. Note that some assets (particularly `keydz.svg`) have specific licensing requirements detailed in the LICENSE file.

## For Developers

If you're interested in the technical implementation, audio processing techniques, or want to adapt this for similar projects, see [CLAUDE.md](CLAUDE.md) for detailed development notes including:

- Audio frequency analysis methodology
- Spectral subtraction techniques
- Web Audio API implementation
- Adaptive video quality system
- Lessons learned from failed approaches

---

*This project demonstrates how art and technology can document and resist oppression, transforming the tools of surveillance into instruments of creative expression.*
