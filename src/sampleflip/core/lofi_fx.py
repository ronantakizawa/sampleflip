"""
lofi_fx.py -- Lo-fi processing, vinyl noise, sub bass synthesis.

Shared effects used by boom-bap and other genres that need vintage character.

Usage:
    from lofi_fx import lofi_process, vinyl_noise, vinyl_crackle,
        parallel_drum_compress, generate_sub_bass
"""

import numpy as np
from scipy.signal import butter, sosfilt
import pedalboard as pb

SR = 44100


def apply_pb(arr2ch, board, sr=SR):
    out = board(arr2ch.T.astype(np.float32), sr)
    return out.T.astype(np.float32)


def lofi_process(audio, sr=SR, intensity=0.5):
    """Lo-fi chain: bit-crush + tape saturation + high-shelf cut."""
    out = audio.copy()
    if intensity > 0.2:
        bits = max(8, int(16 - intensity * 4))
        levels = 2 ** bits
        out = np.round(out * levels) / levels
    drive = 1.0 + intensity * 0.8
    out = np.tanh(out * drive).astype(np.float32)
    cutoff = max(4000, min(int(18000 - intensity * 6000), sr // 2 - 100))
    sos = butter(2, cutoff, btype='low', fs=sr, output='sos')
    return sosfilt(sos, out).astype(np.float32)


def vinyl_noise(n_samples, sr=SR, density=10, amplitude=0.004):
    """Vinyl crackle/dust overlay (boom-bap style, heavier)."""
    rng = np.random.RandomState(42)
    noise = np.zeros(n_samples, dtype=np.float32)
    n_clicks = int(density * n_samples / sr)
    positions = rng.randint(0, n_samples, n_clicks)
    amps = rng.uniform(0.3, 1.0, n_clicks).astype(np.float32) * amplitude
    for pos, amp in zip(positions, amps):
        click_len = rng.randint(2, 8)
        end = min(pos + click_len, n_samples)
        noise[pos:end] += amp * rng.uniform(-1, 1, end - pos).astype(np.float32)
    sos = butter(2, [1000, 8000], btype='bandpass', fs=sr, output='sos')
    return sosfilt(sos, noise).astype(np.float32)


def vinyl_crackle(n_samples, sr=SR, density=6, amplitude=0.003):
    """Light vinyl texture (jazz house style, subtler)."""
    return vinyl_noise(n_samples, sr, density=density, amplitude=amplitude)


def parallel_drum_compress(drum_stereo, sr=SR):
    """70% dry + 30% heavily compressed for tape-style grit."""
    compressed = apply_pb(drum_stereo.copy(), pb.Pedalboard([
        pb.Compressor(threshold_db=-20, ratio=8.0, attack_ms=1, release_ms=50),
        pb.Gain(gain_db=4.0),
    ]), sr)
    compressed = np.tanh(compressed * 1.3).astype(np.float32)
    return (drum_stereo * 0.70 + compressed * 0.30).astype(np.float32)


def generate_sub_bass(freq_hz, duration_s, sr=SR, attack_ms=8, release_ms=60):
    """Pure sine sub bass with ADSR and subtle 2nd harmonic."""
    n = int(duration_s * sr)
    if n < 1:
        return np.zeros(1, dtype=np.float32)
    t = np.arange(n) / sr
    wave = np.sin(2 * np.pi * freq_hz * t).astype(np.float32)
    wave += 0.08 * np.sin(2 * np.pi * freq_hz * 2 * t).astype(np.float32)
    attack = min(int(attack_ms / 1000 * sr), n // 4)
    release = min(int(release_ms / 1000 * sr), n // 3)
    env = np.ones(n, dtype=np.float32) * 0.9
    if attack > 0:
        env[:attack] = np.linspace(0, 0.9, attack)
    if release > 0:
        env[-release:] = np.linspace(0.9, 0, release)
    return (wave * env).astype(np.float32)
