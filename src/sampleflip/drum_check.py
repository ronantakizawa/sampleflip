"""Check if an audio sample contains drums/percussion."""

import numpy as np
import librosa


def has_drums(audio_path, threshold=0.30):
    """Detect if a sample contains drums/percussion.

    Checks:
    1. Onset strength — drums have sharp transients
    2. Spectral flatness — drums are noise-like (flat spectrum), melodies are tonal (peaked)
    3. High-frequency energy ratio — hats/cymbals have lots of energy above 8kHz

    Args:
        audio_path: path to WAV/MP3
        threshold: 0-1, above this = drums detected (default 0.45)

    Returns: (has_drums: bool, score: float, details: str)
    """
    y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=30)

    if len(y) < sr:
        return False, 0.0, 'too short to analyze'

    # 1. Onset strength — sharp transients = drums
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_mean = float(np.mean(onset_env))
    onset_std = float(np.std(onset_env))
    # High std/mean ratio = peaky transients (drums)
    onset_ratio = onset_std / (onset_mean + 1e-9)
    onset_score = min(1.0, onset_ratio / 3.0)  # normalize to 0-1

    # 2. Spectral flatness — noise-like (drums) vs tonal (melody)
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    flatness_mean = float(np.mean(flatness))
    flatness_score = min(1.0, flatness_mean * 5.0)  # 0.2+ flatness = very noisy

    # 3. High-frequency energy ratio — hats/cymbals above 8kHz
    S = np.abs(librosa.stft(y, n_fft=2048))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    total_energy = float(np.sum(S ** 2)) + 1e-9
    hf_energy = float(np.sum(S[freqs > 8000, :] ** 2))
    hf_ratio = hf_energy / total_energy
    hf_score = min(1.0, hf_ratio * 10.0)  # 10%+ HF energy = cymbals/hats

    # Combined score
    score = onset_score * 0.4 + flatness_score * 0.3 + hf_score * 0.3

    details = f'onset={onset_score:.2f} flatness={flatness_score:.2f} hf={hf_score:.2f}'
    return score > threshold, score, details
