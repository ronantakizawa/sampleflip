"""
mix_master.py -- Mastering, LUFS normalization, export, and mix analysis.

Shared pipeline for all renderers: master EQ/compression, loudness
normalization, WAV/MP3 export, per-section metering, artifact checks.

Usage:
    from mix_master import master_and_export, mix_analysis
"""

import os
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import pedalboard as pb
import pyloudnorm as pyln

SR = 44100


def apply_pb(arr2ch, board, sr=SR):
    """Apply pedalboard effects chain to stereo array."""
    out = board(arr2ch.T.astype(np.float32), sr)
    return out.T.astype(np.float32)


def master_chain(mix, sr=SR, hpf_hz=25, lpf_hz=18000,
                 comp_threshold=-14, comp_ratio=2.0,
                 comp_attack_ms=20, comp_release_ms=250, gain_db=-1.0):
    """Apply master EQ + compression + gain."""
    board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=hpf_hz),
        pb.LowpassFilter(cutoff_frequency_hz=lpf_hz),
        pb.Compressor(threshold_db=comp_threshold, ratio=comp_ratio,
                      attack_ms=comp_attack_ms, release_ms=comp_release_ms),
        pb.Gain(gain_db=gain_db),
    ])
    return apply_pb(mix, board, sr)


def fade_out(mix, fade_start_samp, fade_end_samp):
    """Apply quadratic fade-out to stereo mix."""
    fade_len = fade_end_samp - fade_start_samp
    if fade_len <= 0:
        return mix
    fade_curve = np.linspace(1.0, 0.0, fade_len) ** 2
    mix[fade_start_samp:fade_end_samp, 0] *= fade_curve
    mix[fade_start_samp:fade_end_samp, 1] *= fade_curve
    return mix


def lufs_normalize(mix, measure_start, measure_end, sr=SR, target_lufs=-14.0):
    """Normalize mix to target LUFS measured on a reference region.
    Applies soft-clipping if peaks exceed 0.95. Returns modified mix."""
    meter = pyln.Meter(sr, block_size=0.400)
    region = mix[measure_start:measure_end]
    lufs_before = meter.integrated_loudness(region)
    print(f'  Pre-norm LUFS (hook): {lufs_before:.1f}')

    if np.isfinite(lufs_before):
        gain_db = target_lufs - lufs_before
        mix = mix * (10 ** (gain_db / 20.0))
        print(f'  Applied {gain_db:+.1f} dB gain')

    peak = np.abs(mix).max()
    if peak > 0.95:
        print(f'  Peak: {peak:.3f} — soft clipping')
        mix = np.tanh(mix * (1.0 / peak) * 1.2) * 0.95
    else:
        print(f'  Peak: {peak:.3f} — clean')

    lufs_after = meter.integrated_loudness(mix[measure_start:min(measure_end, len(mix))])
    print(f'  Final LUFS (hook): {lufs_after:.1f}  (target: {target_lufs})')
    return mix


def export_audio(mix, out_wav, out_mp3, sr=SR, tags=None):
    """Export stereo mix to WAV + MP3. Returns MP3 file size string."""
    out_i16 = (mix * 32767).clip(-32767, 32767).astype(np.int16)
    wavfile.write(out_wav, sr, out_i16)
    seg = AudioSegment.from_wav(out_wav)
    seg.export(out_mp3, format='mp3', bitrate='192k', tags=tags or {})
    m, s = divmod(int(len(seg) / 1000), 60)
    size_mb = os.path.getsize(out_mp3) / 1e6
    print(f'  {os.path.basename(out_mp3)}: {size_mb:.1f} MB  |  {m}:{s:02d}')
    return out_mp3


def mix_analysis(mix, name, sections, b2s_func, sr=SR):
    """Print per-section LUFS/peak metering and run artifact checks.

    Args:
        mix: stereo (N,2) array
        name: track name
        sections: list of (section_name, start_bar, end_bar)
        b2s_func: function(bar) -> seconds
    """
    meter = pyln.Meter(sr, block_size=0.400)
    y_mono = mix.mean(axis=1).astype(np.float32)
    rms_val = np.sqrt(np.mean(y_mono ** 2))
    final_lufs = meter.integrated_loudness(mix)
    peak_db = 20 * np.log10(np.abs(mix).max() + 1e-9)

    print(f'\n== Mix Analysis: {name} ==')
    print(f'  LUFS: {final_lufs:.1f}  |  Peak: {peak_db:.1f} dB  |  '
          f'RMS: {20*np.log10(rms_val+1e-9):.1f} dB')

    for sec_name, s_bar, e_bar in sections:
        s_samp = int(b2s_func(s_bar) * sr)
        e_samp = min(int(b2s_func(e_bar) * sr), len(mix))
        section = mix[s_samp:e_samp]
        if len(section) < sr:
            continue
        sec_lufs = meter.integrated_loudness(section)
        sec_peak = 20 * np.log10(np.abs(section).max() + 1e-9)
        print(f'  {sec_name:<10} LUFS: {sec_lufs:>6.1f}  peak: {sec_peak:>6.1f} dB')

    print('\n-- Artifact Check --')
    clip_count = np.sum(np.abs(mix) > 0.99)
    print(f'  Clipping: {"none" if clip_count == 0 else f"{clip_count} samples"}')
    dc_L = np.abs(np.mean(mix[:, 0]))
    dc_R = np.abs(np.mean(mix[:, 1]))
    print(f'  DC offset: {"clean" if dc_L < 0.005 and dc_R < 0.005 else f"L={dc_L:.4f} R={dc_R:.4f}"}')
    mono_rms = np.sqrt(np.mean(y_mono ** 2))
    stereo_rms = np.sqrt(np.mean(mix ** 2))
    mono_loss = 20 * np.log10(mono_rms / (stereo_rms + 1e-9) + 1e-9)
    print(f'  Phase: {mono_loss:.1f} dB mono loss {"(ok)" if mono_loss >= -3.0 else "<< CHECK"}')


def get_version(output_dir, name):
    """Auto-increment version number by scanning existing files."""
    import glob
    existing = glob.glob(os.path.join(output_dir, f'{name}_v*.mp3'))
    ver = max([int(f.split('_v')[-1].split('.')[0]) for f in existing], default=0) + 1
    return ver


def master_and_export(mix, name, output_dir, sections, b2s_func,
                      song_dur_samp, nbars, sr=SR,
                      hpf_hz=25, lpf_hz=18000,
                      comp_threshold=-14, comp_ratio=2.0,
                      target_lufs=-14.0,
                      hook_start_bar=8, hook_end_bar=24,
                      genre='Hip-Hop', album='Beats',
                      fade_bars=4):
    """Full master-to-export pipeline. Returns path to MP3."""
    ver = get_version(output_dir, name)
    vstr = f'v{ver}'
    out_wav = os.path.join(output_dir, f'{name}_{vstr}.wav')
    out_mp3 = os.path.join(output_dir, f'{name}_{vstr}.mp3')

    # Master chain
    print(f'\nMaster chain ...')
    mix = master_chain(mix, sr, hpf_hz=hpf_hz, lpf_hz=lpf_hz,
                       comp_threshold=comp_threshold, comp_ratio=comp_ratio)

    # Trim
    trim = min(song_dur_samp, len(mix))
    mix = mix[:trim]

    # Fade out
    fade_start = int(b2s_func(nbars - fade_bars) * sr)
    mix = fade_out(mix, fade_start, trim)

    # LUFS normalize
    print(f'\nLUFS normalization ...')
    measure_start = int(b2s_func(hook_start_bar) * sr)
    measure_end = int(b2s_func(hook_end_bar) * sr)
    mix = lufs_normalize(mix, measure_start, measure_end, sr, target_lufs)

    # Export
    print(f'\nExporting ...')
    export_audio(mix, out_wav, out_mp3, sr, tags={
        'title': f'{name} {vstr}',
        'artist': 'Claude Code',
        'album': album,
        'genre': genre,
    })

    # Analysis
    mix_analysis(mix, name, sections, b2s_func, sr)

    print(f'\nDone!  ->  {out_mp3}')
    return out_mp3
