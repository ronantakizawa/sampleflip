"""
gross_beat.py -- Gross Beat-style time/volume manipulation effects.

Inspired by FL Studio's Gross Beat plugin. Applied to the sample channel
only (drums play through unaffected). Effects are randomized per track
using a seeded RNG, placed only at transitions.

Usage:
    from gross_beat import apply_gross_beat
"""

import numpy as np


def gb_reverse(audio, sr):
    """Reverse playback with fade edges."""
    out = audio[::-1].copy()
    xf = min(int(0.003 * sr), len(out) // 8)
    if xf > 0:
        out[:xf] *= np.linspace(0, 1, xf).astype(np.float32)
        out[-xf:] *= np.linspace(1, 0, xf).astype(np.float32)
    return out


def gb_stutter(audio, sr, divisions=8, wet=0.5):
    """Rapid repeat of a small slice. wet controls dry/wet blend."""
    slice_len = max(1, len(audio) // divisions)
    slc = audio[:slice_len].copy()
    fade = min(int(0.002 * sr), slice_len // 4)
    if fade > 0:
        slc[-fade:] *= np.linspace(1, 0, fade).astype(np.float32)
    stuttered = np.tile(slc, divisions + 1)[:len(audio)].astype(np.float32)
    return (audio * (1 - wet) + stuttered * wet).astype(np.float32)


def gb_gate(audio, sr, rate=8, wet=0.5):
    """Rhythmic volume gating. wet controls dry/wet blend."""
    n = len(audio)
    gate_len = max(1, n // rate)
    env = np.ones(n, dtype=np.float32)
    fade = min(int(0.002 * sr), gate_len // 4)
    for i in range(rate):
        on_start = i * gate_len
        off_start = on_start + gate_len // 2
        off_end = min(on_start + gate_len, n)
        if off_start < n:
            env[off_start:off_end] = 0.0
            if fade > 0 and off_start + fade <= n:
                env[off_start:off_start + fade] = np.linspace(1, 0, fade).astype(np.float32)
            if fade > 0 and off_end - fade >= 0 and off_end <= n:
                actual_fade = min(fade, off_end - max(off_start, off_end - fade))
                if actual_fade > 0:
                    env[off_end - actual_fade:off_end] = np.linspace(0, 1, actual_fade).astype(np.float32)
    gated = (audio * env).astype(np.float32)
    return (audio * (1 - wet) + gated * wet).astype(np.float32)


def gb_underwater(audio, sr, cutoff=800, wet=0.85):
    """Low-pass filter effect — muffled/underwater sound."""
    from scipy.signal import butter, sosfilt
    sos = butter(4, cutoff, btype='low', fs=sr, output='sos')
    filtered = sosfilt(sos, audio).astype(np.float32)
    return (audio * (1 - wet) + filtered * wet).astype(np.float32)


# Default FX pool for transition points
DEFAULT_FX_POOL = [
    ('underwater', gb_underwater, {'beat_offset': 0, 'n_beats': 4, 'cutoff': 800}),
    ('underwater', gb_underwater, {'beat_offset': 0, 'n_beats': 4, 'cutoff': 600}),
    ('underwater', gb_underwater, {'beat_offset': 0, 'n_beats': 2, 'cutoff': 1000}),
    ('reverse',  gb_reverse,  {'beat_offset': 2, 'n_beats': 2}),
    ('stutter4', gb_stutter,  {'beat_offset': 3, 'n_beats': 1, 'divisions': 4}),
    ('stutter6', gb_stutter,  {'beat_offset': 3, 'n_beats': 1, 'divisions': 6}),
    ('stutter8', gb_stutter,  {'beat_offset': 2, 'n_beats': 2, 'divisions': 8}),
    ('gate',     gb_gate,     {'beat_offset': 0, 'n_beats': 4, 'rate': 8}),
]


def apply_gross_beat(samp_out, sr, bar_dur, nbars, transition_bars,
                     track_name='', fx_pool=None):
    """Apply Gross Beat-style FX at transition bar positions.

    Args:
        samp_out: mono sample array (modified in-place)
        sr: sample rate
        bar_dur: duration of one bar in seconds
        nbars: total number of bars
        transition_bars: list of bar numbers where effects can be placed
        track_name: seed for deterministic randomization
        fx_pool: list of (name, func, config) tuples. Uses DEFAULT_FX_POOL if None.

    Returns:
        (samp_out, log) where log is a list of strings describing applied effects.
    """
    if fx_pool is None:
        fx_pool = DEFAULT_FX_POOL

    beat_samp = int((bar_dur / 4) * sr)
    bar_samp = int(bar_dur * sr)
    n = len(samp_out)
    log = []
    rng = np.random.RandomState(hash(track_name) % (2**31))

    def apply_at(bar, beat_offset, n_beats, fx_func, fx_name, **kwargs):
        start = int(bar * bar_samp + beat_offset * beat_samp)
        end = min(start + n_beats * beat_samp, n)
        if start >= n or end <= start:
            return
        seg = samp_out[start:end].copy()
        processed = fx_func(seg, sr, **kwargs) if kwargs else fx_func(seg, sr)
        if len(processed) >= end - start:
            samp_out[start:end] = processed[:end - start]
        else:
            samp_out[start:start + len(processed)] = processed
        log.append(f'    bar {bar}+{beat_offset}: {fx_name} ({n_beats} beats)')

    # Randomly decide how many transitions get an effect
    n_fx = rng.randint(1, len(transition_bars) + 1)
    chosen = sorted(rng.choice(len(transition_bars), size=n_fx, replace=False))

    for idx in chosen:
        bar = transition_bars[idx]
        fx_name, fx_func, cfg = fx_pool[rng.randint(len(fx_pool))]
        beat_offset = cfg['beat_offset']
        n_beats = cfg['n_beats']
        kwargs = {k: v for k, v in cfg.items() if k not in ('beat_offset', 'n_beats')}
        apply_at(bar, beat_offset, n_beats, fx_func, fx_name, **kwargs)

    return samp_out, log
