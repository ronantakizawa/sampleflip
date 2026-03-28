# SampleFlip

Describe a beat, get a beat. Uses AI to search YouTube for the right sample, then generates a full beat with drums, bass, arrangement, and mastering.

## Install

```bash
pip install sampleflip
```

**System requirements:**
```bash
brew install yt-dlp ffmpeg
```

**Set your API key:**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

```bash
# Just describe what you want
sampleflip "dark trap beat with piano"
sampleflip "jazzy house with sax and warm pads"
sampleflip "aggressive drill beat with orchestral strings"
sampleflip "chill lo-fi boom bap with vinyl samples"
sampleflip "melodic trap like Gunna with guitar"

# Override genre or BPM if needed
sampleflip "emotional beat" --genre rnb --bpm 75
```

## How It Works

1. **AI plans the beat** — Claude reads your description, picks the genre, BPM, and crafts a YouTube search query for the right sample
2. **Searches YouTube** — finds sample packs, loop kits, and melody loops matching your vibe
3. **Downloads** the best result as WAV
4. **Analyzes** the sample: BPM, key, chords, spectral character
5. **Extracts** the best loop (scored by harmonic richness + rhythmic interest)
6. **Programs drums** from genre-specific patterns
7. **Generates bass** following detected chord progression
8. **Arranges** into full song structure with transitions and FX
9. **Masters** to -14 LUFS
10. **Exports** MP3 + WAV to `./output/`

## Options

```
sampleflip <prompt> [options]

Options:
  --genre CHOICE    Override genre (trap, boombap, jazzhouse, rnb, drill, etc.)
  --bpm FLOAT       Override BPM
  --bars INT        Number of bars
  --drums PATH      Custom drum pattern JSON
  --loop-start      Loop start in seconds
  --loop-end        Loop end in seconds
  --vinyl-slow      Pitch+speed linked slowdown
  --no-bass         Disable bass
  --lofi FLOAT      Lo-fi intensity (0-1)
  --output PATH     Output directory (default: ./output)
  --kit PATH        Drum kit samples directory
```

## Supported Genres

| Genre | BPM Range | Style |
|-------|-----------|-------|
| trap | 140-160 | Dark 808s, rolling hats, half-time clap |
| boombap | 75-95 | Vinyl chops, drum breaks, lo-fi |
| jazzhouse | 120-128 | Four-on-the-floor, ride, warm pads |
| progressive_house | 126-132 | Big drops, filter sweeps |
| rnb | 65-85 | Soft drums, jazz chords, swing |
| drill | 138-144 | 808 slides, dark piano, aggressive hats |
| melodic_trap | 135-150 | Emotional melodies, bouncy 808 |
| 2hollis | 140-160 | Hyperpop, distorted, reese bass |
| techno | 128-140 | Four-on-floor, acid, mechanical hats |
| breakcore | 160-180 | Chopped amen breaks, reese bass |
