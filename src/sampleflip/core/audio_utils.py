"""
audio_utils.py -- Shared audio utilities for beat production.

Core functions used across all renderers: sample loading, placement,
effects processing, stereo manipulation, loop creation, and filtering.

Usage:
    from audio_utils import load_sample, place, apply_pb, create_sample_bed
"""

import numpy as np
from math import gcd
from scipy import signal
from scipy.signal import butter, sosfilt
import soundfile as sf
import pedalboard as pb

SR = 44100


def load_sample(path, sr=SR):
    """Load audio file as mono float32 at target sample rate."""
    data, orig_sr = sf.read(path, dtype='float32', always_2d=True)
    mono = data.mean(axis=1)
    if orig_sr != sr:
        g = gcd(sr, orig_sr)
        mono = signal.resample_poly(mono, sr // g, orig_sr // g)
    return mono.astype(np.float32)


def place(buf_L, buf_R, snd, start_s, gain_L=1.0, gain_R=1.0):
    """Place a mono sound into stereo buffers at a sample position."""
    nsamp = len(buf_L)
    e = min(start_s + len(snd), nsamp)
    if e <= start_s or start_s < 0:
        return
    chunk = snd[:e - start_s]
    buf_L[start_s:e] += chunk * gain_L
    buf_R[start_s:e] += chunk * gain_R


def apply_pb(arr2ch, board, sr=SR):
    """Apply a pedalboard effects chain to a stereo (N,2) array."""
    out = board(arr2ch.T.astype(np.float32), sr)
    return out.T.astype(np.float32)


def bar_to_s(bar, beat=0.0, bar_dur=None):
    """Convert bar + beat position to seconds."""
    return (bar + beat / 4.0) * bar_dur


def pan_stereo(buf, position):
    """Pan a stereo buffer. position: -1 (left) to +1 (right)."""
    angle = (position + 1) * np.pi / 4
    result = buf.copy()
    result[:, 0] *= np.cos(angle)
    result[:, 1] *= np.sin(angle)
    return result


def stereo_widen(buf, delay_ms=12, sr=SR):
    """Widen stereo image via Haas-effect delay on right channel."""
    d = int(delay_ms / 1000.0 * sr)
    if d <= 0 or d >= len(buf):
        return buf.copy()
    result = buf.copy()
    result[d:, 1] = buf[:-d, 1]
    result[:d, 1] *= 0.3
    return result


def create_sample_bed(loop, total_samp, crossfade_ms=30, sr=SR):
    """Tile a loop to fill total_samp with crossfades at joins."""
    xf = int(crossfade_ms / 1000.0 * sr)
    xf = min(xf, len(loop) // 4)
    bed = np.zeros(total_samp, dtype=np.float32)
    pos = 0
    while pos < total_samp:
        end = min(pos + len(loop), total_samp)
        bed[pos:end] += loop[:end - pos]
        if pos > 0 and xf > 0:
            cf_end = min(pos + xf, total_samp)
            cf_len = cf_end - pos
            bed[pos:cf_end] *= np.linspace(0, 1, cf_len).astype(np.float32)
            fade_start = max(pos - xf, 0)
            fade_len = pos - fade_start
            if fade_len > 0:
                bed[fade_start:pos] *= np.linspace(1, 0, fade_len).astype(np.float32)
        pos += len(loop)
    return bed


def lpf_sweep(audio, sr, start_cutoff, end_cutoff, n_blocks=64):
    """Block-wise LPF sweep from start_cutoff to end_cutoff Hz."""
    block_size = len(audio) // n_blocks
    if block_size < 256:
        return audio.copy()
    out = np.zeros_like(audio)
    cutoffs = np.linspace(start_cutoff, end_cutoff, n_blocks)
    zi = None
    for i in range(n_blocks):
        s = i * block_size
        e = s + block_size if i < n_blocks - 1 else len(audio)
        cutoff = np.clip(cutoffs[i], 60, sr // 2 - 100)
        sos = butter(4, cutoff, btype='low', fs=sr, output='sos')
        if zi is None:
            block_out, zi = sosfilt(sos, audio[s:e], zi=np.zeros((sos.shape[0], 2)))
        else:
            block_out, zi = sosfilt(sos, audio[s:e], zi=zi)
        out[s:e] = block_out
    return out.astype(np.float32)


def pitch_shift_sample(sample, semitones):
    """Pitch shift a sample by resampling (changes duration)."""
    if abs(semitones) < 0.01:
        return sample.copy()
    ratio = 2.0 ** (semitones / 12.0)
    new_len = int(len(sample) / ratio)
    if new_len < 10:
        return sample.copy()
    return signal.resample(sample, new_len).astype(np.float32)


def auto_gain_sample(sample_bed, target_rms_db=-18.0):
    """Normalize sample bed to target RMS level. Returns (gained_bed, gain_db)."""
    target_rms = 10 ** (target_rms_db / 20.0)
    bed_rms = np.sqrt(np.mean(sample_bed ** 2))
    if bed_rms < 1e-6:
        return sample_bed, 0.0
    gain = target_rms / bed_rms
    gain_db = 20 * np.log10(gain)
    return (sample_bed * gain).astype(np.float32), gain_db


def adaptive_hpf(loop, sr=SR, baseline_hz=150, max_hz=220):
    """Measure low-end energy ratio and return appropriate HPF cutoff.
    Bass-heavy samples get a higher cutoff to make room for 808/sub."""
    spec = np.abs(np.fft.rfft(loop))
    freqs = np.fft.rfftfreq(len(loop), 1.0 / sr)
    low_energy = float(spec[freqs < 300].sum())
    total_energy = float(spec.sum()) + 1e-9
    low_ratio = low_energy / total_energy
    hpf_cutoff = int(baseline_hz + max(0, low_ratio - 0.20) * (max_hz - baseline_hz) / 0.30)
    hpf_cutoff = min(hpf_cutoff, max_hz)
    return hpf_cutoff, low_ratio


def add_metronome(mix, loop_len, n_loop_bars, nsamp, sr=SR):
    """Add metronome clicks to a stereo mix array. Returns modified mix."""
    click_dur = 0.015
    click_len = int(click_dur * sr)
    t_click = np.linspace(0, click_dur, click_len, endpoint=False).astype(np.float32)
    click_hi = (0.35 * np.sin(2 * np.pi * 1500 * t_click) * np.exp(-t_click * 200)).astype(np.float32)
    click_lo = (0.20 * np.sin(2 * np.pi * 1000 * t_click) * np.exp(-t_click * 200)).astype(np.float32)
    beat_samp_interval = loop_len / (n_loop_bars * 4)
    total_beats = int(nsamp / beat_samp_interval)
    for b in range(total_beats):
        pos = int(b * beat_samp_interval)
        if pos + click_len > nsamp:
            break
        click = click_hi if b % 4 == 0 else click_lo
        mix[pos:pos + click_len, 0] += click
        mix[pos:pos + click_len, 1] += click
    return mix
