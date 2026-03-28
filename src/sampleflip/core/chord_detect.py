"""
chord_detect.py -- Detect chord progression from a sample loop.

Uses template-based chord recognition with cosine similarity:
  1. Define chord templates (major, minor triads as binary pitch-class vectors)
  2. For each bar, compute chroma vector
  3. Cosine similarity against all 24 chord templates (12 roots x 2 qualities)
  4. Bonus for chords diatonic to the detected key
  5. Pick best match per bar

Key detection uses Krumhansl-Schmuckler (in sample_analysis.py).

Usage:
    from chord_detect import detect_chords, chords_to_bass_pattern
    chords = detect_chords(loop, sr=44100, n_bars=4, key='Em')
"""

import numpy as np
import librosa

SR = 44100
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# ============================================================================
# Chord templates: binary pitch-class vectors rotated for each root
# ============================================================================

# Base templates (root = C, index 0)
CHORD_TEMPLATES = {
    'major': np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32),  # C E G
    'minor': np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32),  # C Eb G
}


def _rotate(template, semitones):
    """Rotate a pitch-class template by N semitones."""
    return np.roll(template, semitones)


def _cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ============================================================================
# Diatonic chord tables for key-constraint scoring
# ============================================================================

# Major key: I ii iii IV V vi vii°
# e.g. C major: C Dm Em F G Am Bdim
MAJOR_DIATONIC = {
    0: 'major',  # I
    2: 'minor',  # ii
    4: 'minor',  # iii
    5: 'major',  # IV
    7: 'major',  # V
    9: 'minor',  # vi
    11: 'minor', # vii (simplified from dim)
}

# Minor key: i ii° III iv v VI VII
# e.g. A minor: Am Bdim C Dm Em F G
MINOR_DIATONIC = {
    0: 'minor',  # i
    2: 'minor',  # ii (simplified from dim)
    3: 'major',  # III
    5: 'minor',  # iv
    7: 'minor',  # v
    8: 'major',  # VI
    10: 'major', # VII
}


def _parse_key(key_str):
    """Parse key string like 'C', 'Em', 'F#m' into (root_pc, is_minor)."""
    if key_str is None:
        return None, False
    key_str = key_str.strip()
    is_minor = key_str.endswith('m')
    root_name = key_str.rstrip('m').strip()
    if root_name in NOTE_NAMES:
        return NOTE_NAMES.index(root_name), is_minor
    return None, False


def _is_diatonic(root_pc, quality, key_pc, key_minor):
    """Check if a chord is diatonic to the given key."""
    diatonic = MINOR_DIATONIC if key_minor else MAJOR_DIATONIC
    interval = (root_pc - key_pc) % 12
    if interval in diatonic:
        return diatonic[interval] == quality
    return False


# ============================================================================
# Main chord detection
# ============================================================================

def detect_chords(loop, sr=SR, n_bars=2, key=None):
    """Detect chord root and quality for each bar using template matching.

    Args:
        loop: mono audio array
        sr: sample rate
        n_bars: number of bars in the loop
        key: detected key string (e.g. 'B', 'Em', 'F#m') for diatonic bonus

    Returns:
        list of dicts per bar: {'root_midi', 'root_name', 'quality', 'root_pc'}
        root_midi is in octave 2 (C2=36) for bass use.
    """
    bar_len = len(loop) // n_bars
    key_pc, key_minor = _parse_key(key)

    chords = []
    for i in range(n_bars):
        seg = loop[i * bar_len:(i + 1) * bar_len]
        if len(seg) < 512:
            chords.append({'root_midi': 36, 'root_name': 'C',
                           'quality': 'minor', 'root_pc': 0})
            continue

        chroma = librosa.feature.chroma_cqt(y=seg, sr=sr, hop_length=512)
        chroma_avg = chroma.mean(axis=1)

        # Normalize chroma for cosine similarity
        chroma_norm = chroma_avg / (np.linalg.norm(chroma_avg) + 1e-9)

        # Score all 24 chords (12 roots x 2 qualities)
        best_score = -1.0
        best_root = 0
        best_quality = 'major'

        for root_pc in range(12):
            for quality, template in CHORD_TEMPLATES.items():
                rotated = _rotate(template, root_pc)
                score = _cosine_sim(chroma_norm, rotated)

                # Diatonic bonus: +15% for in-key chords
                if key_pc is not None and _is_diatonic(root_pc, quality, key_pc, key_minor):
                    score *= 1.15

                if score > best_score:
                    best_score = score
                    best_root = root_pc
                    best_quality = quality

        root_name = NOTE_NAMES[best_root]

        # Bass MIDI note in octave 2 (C2=36)
        root_midi = 36 + best_root
        # Drop to octave 1 if root is high (G#, A, A#, B) for deeper bass
        if best_root >= 8:
            root_midi -= 12

        chords.append({
            'root_midi': root_midi,
            'root_name': root_name,
            'quality': best_quality,
            'root_pc': best_root,
        })

    return chords


def chords_to_bass_pattern(chords, n_bars_song):
    """Tile detected chord roots across the full song length.
    Returns list of root_midi values, one per bar."""
    if not chords:
        return [36] * n_bars_song
    pattern = []
    for i in range(n_bars_song):
        chord = chords[i % len(chords)]
        pattern.append(chord['root_midi'])
    return pattern
