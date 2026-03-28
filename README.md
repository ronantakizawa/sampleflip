# SampleFlip

Search YouTube for samples, generate full beats with drums, bass, arrangement, and mastering.

## Install

```bash
pip install sampleflip
```

**System requirements:**
```bash
brew install yt-dlp ffmpeg
```

## Usage

### Search for samples
```bash
sampleflip search "dark piano melody loop"
```

### Generate a beat from a result
```bash
sampleflip generate 1 --genre trap --bpm 145
```

### All options
```bash
sampleflip generate <number> --genre <genre> [options]

Options:
  --bpm FLOAT           Override BPM (auto-detected if omitted)
  --bars INT            Number of bars
  --drums PATH          Custom drum pattern JSON
  --loop-start FLOAT    Loop start in seconds
  --loop-end FLOAT      Loop end in seconds
  --vinyl-slow          Pitch+speed linked slowdown (vinyl effect)
  --no-bass             Disable bass
  --lofi FLOAT          Lo-fi intensity (0-1)
  --output PATH         Output directory (default: ./output)
  --kit PATH            Drum kit samples directory
  --name TEXT            Beat name
```

### List genres
```bash
sampleflip genres
```

### List drum presets
```bash
sampleflip presets
```

## Supported Genres

| Genre | BPM Range | Style |
|-------|-----------|-------|
| trap | 120-165 | Dark 808s, rolling hats, half-time clap |
| boombap | 70-100 | Vinyl chops, drum breaks, lo-fi |
| jazzhouse | 115-135 | Four-on-the-floor, ride, warm pads |
| progressive_house | 118-135 | Big drops, filter sweeps, saw leads |
| rnb | 60-95 | Soft drums, jazz chords, swing |
| drill | 130-150 | 808 slides, dark piano, aggressive hats |
| melodic_trap | 130-150 | Emotional melodies, bouncy 808 |
| 2hollis | 130-165 | Hyperpop, distorted, reese bass |
| techno | 124-145 | Four-on-floor, acid, mechanical hats |
| breakcore | 85-120 | Chopped amen breaks, reese bass |

## How It Works

1. **Search** YouTube via yt-dlp
2. **Download** selected result as WAV
3. **Analyze** sample: BPM, key, chords, spectral character
4. **Extract** best loop (quality-scored: harmonic richness + rhythmic interest)
5. **Align** tempo to genre range
6. **Program** drums from MIDI event patterns (genre-specific or custom JSON)
7. **Generate** bass following detected chord progression
8. **Arrange** into full song structure with transitions/FX
9. **Master** to -14 LUFS with compression and limiting
10. **Export** MP3 + WAV
