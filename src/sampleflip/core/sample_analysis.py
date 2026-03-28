"""
sample_analysis.py -- Sample analysis and loop extraction utilities.

Uses madmom (CRNN neural network) for beat/tempo detection with librosa
as fallback. Madmom is significantly more accurate than librosa's
onset-correlation method for diverse audio.

Usage:
    from sample_analysis import detect_sample_tempo, detect_sample_key,
        detect_loop_period, extract_loop_auto, extract_loop_at
"""

import numpy as np
import librosa
import tempfile, os
from scipy.io import wavfile

SR = 44100

# Monkey-patch numpy for madmom compatibility (Python 3.11+)
np.float = float
np.int = int
np.bool = bool
np.str = str
np.complex = complex
np.object = object

try:
    import madmom
    HAVE_MADMOM = True
except ImportError:
    HAVE_MADMOM = False


def _madmom_beats(audio, sr=SR):
    """Get beat positions using madmom's RNN beat tracker.
    Returns beat times in seconds."""
    # Madmom needs a file path, so write temp wav
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        tmp_path = f.name
        wav_data = (audio * 32767).clip(-32767, 32767).astype(np.int16)
        wavfile.write(tmp_path, sr, wav_data)
    try:
        proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
        act = madmom.features.beats.RNNBeatProcessor()(tmp_path)
        beats = proc(act)
        return beats
    finally:
        os.unlink(tmp_path)


def _madmom_tempo(audio, sr=SR):
    """Get tempo using madmom. Returns BPM float."""
    beats = _madmom_beats(audio, sr)
    if len(beats) < 2:
        return None
    intervals = np.diff(beats)
    intervals = intervals[(intervals > 0.2) & (intervals < 2.0)]  # filter outliers
    if len(intervals) < 2:
        return None
    return float(60.0 / np.median(intervals))


def analyze_sample_character(audio, sr=SR):
    """Spectral analysis: centroid, warmth, brightness, onset rate, etc."""
    centroid = float(librosa.feature.spectral_centroid(y=audio, sr=sr).mean())
    bandwidth = float(librosa.feature.spectral_bandwidth(y=audio, sr=sr).mean())
    spec = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    low_energy = float(spec[freqs < 400].sum())
    high_energy = float(spec[freqs > 4000].sum())
    mid_energy = float(spec[(freqs >= 400) & (freqs <= 4000)].sum())
    total_energy = low_energy + mid_energy + high_energy + 1e-9
    warmth = low_energy / total_energy
    brightness = high_energy / total_energy
    mid_presence = mid_energy / total_energy
    onsets = librosa.onset.onset_detect(y=audio, sr=sr, units='time')
    onset_rate = len(onsets) / (len(audio) / sr) if len(audio) > 0 else 0
    flatness = float(librosa.feature.spectral_flatness(y=audio).mean())
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return {
        'centroid': centroid, 'bandwidth': bandwidth,
        'warmth': warmth, 'brightness': brightness,
        'mid_presence': mid_presence, 'onset_rate': onset_rate,
        'flatness': flatness, 'rms': rms,
    }


def detect_vocals(audio, sr=SR):
    """Detect whether a sample contains vocals/speech/singing.

    Uses Demucs (Meta's source separation model) to extract the vocal
    stem, then measures its energy relative to the full mix. If the
    vocal stem has significant energy, the sample has vocals.

    This is the gold standard — Demucs literally separates the voice
    from the instruments. No heuristics needed.

    Returns: (has_vocals: bool, confidence: float 0-1)
    """
    import torch
    import torchaudio
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    # Load model once
    model = get_model('htdemucs')
    model.eval()
    model_sr = model.samplerate

    # Sample 3 positions (25%, 50%, 75%) — 15s each
    chunk_dur = 15  # seconds
    chunk_samples = min(len(audio), int(chunk_dur * sr))
    positions = [0.15, 0.35, 0.55, 0.75] if len(audio) > chunk_samples * 3 else [0.50]

    best_vocal_ratio = 0.0

    for pos_frac in positions:
        start = max(0, int(len(audio) * pos_frac) - chunk_samples // 2)
        start = min(start, max(0, len(audio) - chunk_samples))
        chunk = audio[start:start + chunk_samples]

        # Resample if needed
        if sr != model_sr:
            chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
            chunk = torchaudio.functional.resample(chunk_tensor, sr, model_sr).squeeze(0).numpy()

        # Stereo (batch=1, channels=2, samples)
        waveform = torch.from_numpy(np.stack([chunk, chunk])).float().unsqueeze(0)

        with torch.no_grad():
            sources = apply_model(model, waveform, device='cpu')

        # htdemucs sources: drums=0, bass=1, other=2, vocals=3
        vocals = sources[0, 3].numpy()
        full_mix = waveform[0].numpy()

        vocal_rms = float(np.sqrt(np.mean(vocals ** 2)))
        mix_rms = float(np.sqrt(np.mean(full_mix ** 2))) + 1e-9
        vocal_ratio = vocal_rms / mix_rms
        best_vocal_ratio = max(best_vocal_ratio, vocal_ratio)

    # vocal_ratio > 0.15 = vocals present
    # Pure instrumentals: ratio < 0.08
    # Songs with vocals: ratio 0.20-0.60
    confidence = min(1.0, max(0.0, (best_vocal_ratio - 0.08) / 0.25))
    has_vocals = best_vocal_ratio > 0.15

    return has_vocals, confidence


def detect_sample_tempo(audio, sr=SR, target_bpm=90.0, bpm_range=(70, 200)):
    """Detect tempo using madmom (primary) + librosa (fallback).
    Applies half/double compensation toward target BPM."""
    raw_tempos = []

    # Madmom (neural network — more accurate)
    if HAVE_MADMOM:
        try:
            madmom_bpm = _madmom_tempo(audio, sr)
            if madmom_bpm:
                raw_tempos.append(madmom_bpm)
                print(f'    madmom: {madmom_bpm:.1f} BPM')
        except Exception as e:
            print(f'    madmom failed: {e}')

    # Librosa (fallback)
    tempo_default = float(np.atleast_1d(librosa.beat.tempo(y=audio, sr=sr))[0])
    tempo_hinted = float(np.atleast_1d(
        librosa.beat.tempo(y=audio, sr=sr, start_bpm=target_bpm))[0])
    raw_tempos.extend([tempo_default, tempo_hinted])

    # Try half/double of all detected tempos, pick closest to target
    candidates = []
    for t in raw_tempos:
        for mult in [0.5, 1.0, 2.0]:
            adj = t * mult
            if bpm_range[0] < adj < bpm_range[1]:
                candidates.append((abs(adj - target_bpm), adj, t))
    if candidates:
        candidates.sort()
        return candidates[0][2]
    return tempo_default


def detect_sample_key(audio, sr=SR):
    """Detect key using Krumhansl-Schmuckler algorithm.

    Correlates chroma energy distribution against major/minor key profiles
    for all 24 keys. Returns the key with the highest Pearson correlation.

    Profiles from Krumhansl & Kessler (1982), as used in the K-S algorithm.
    Much more accurate than simple chroma argmax.
    """
    from scipy.stats import zscore as _zscore
    from scipy.linalg import circulant as _circulant

    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
    chroma_dist = chroma.mean(axis=1)  # (12,) pitch class distribution

    if chroma_dist.sum() < 1e-9:
        return 'C'

    x = _zscore(chroma_dist)

    # Krumhansl-Kessler key profiles (C major / C minor reference)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                              2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                              2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    major_z = _zscore(major_profile)
    minor_z = _zscore(minor_profile)

    # Circulant matrix gives all 12 rotations (C, C#, D, ... B)
    major_scores = _circulant(major_z).T.dot(x)  # (12,)
    minor_scores = _circulant(minor_z).T.dot(x)  # (12,)

    key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    best_major_idx = int(np.argmax(major_scores))
    best_minor_idx = int(np.argmax(minor_scores))

    if major_scores[best_major_idx] >= minor_scores[best_minor_idx]:
        return key_names[best_major_idx]
    else:
        return key_names[best_minor_idx] + 'm'


def detect_loop_period(audio, sr=SR):
    """Detect loop repetition period using chroma autocorrelation.
    Returns period in seconds (at least 2.5s)."""
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=512)
    hop_sec = 512 / sr
    n_frames = chroma.shape[1]
    max_lag = min(n_frames // 2, int(25 / hop_sec))
    min_lag = int(2.5 / hop_sec)
    corrs = []
    for lag in range(min_lag, max_lag):
        c1 = chroma[:, :n_frames - lag]
        c2 = chroma[:, lag:]
        sim = np.sum(c1 * c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-9)
        corrs.append((lag * hop_sec, sim))
    corrs.sort(key=lambda x: -x[1])
    return corrs[0][0]


# ============================================================================
# Beat-aligned multi-feature loop finder
# Inspired by librosa_loopfinder: beat-track first, compute feature vectors at
# each beat boundary, then find beat pairs with minimal feature distance whose
# gap matches the target bar count.
# Features: chroma (12), spectral contrast (7), onset strength (1), RMS (1).
# Distance: L1 (Manhattan) — robust to outliers, fast.
# ============================================================================

def _get_beats(audio, sr=SR, target_bpm=None):
    """Get beat times using madmom (primary) or librosa (fallback)."""
    beats_sec = None
    if HAVE_MADMOM:
        try:
            beats_sec = _madmom_beats(audio, sr)
            if beats_sec is not None and len(beats_sec) >= 4:
                print(f'    madmom beats: {len(beats_sec)} ({beats_sec[:4]}...)')
        except Exception:
            beats_sec = None
    if beats_sec is None or len(beats_sec) < 4:
        start_bpm = target_bpm if target_bpm else 120
        _, beat_frames = librosa.beat.beat_track(y=audio, sr=sr,
                                                  start_bpm=start_bpm,
                                                  tightness=400)
        beats_sec = librosa.frames_to_time(beat_frames, sr=sr)
        print(f'    librosa beats: {len(beats_sec)}')
    return np.array(beats_sec, dtype=np.float64)


def _beat_features(audio, sr, beats_sec, hop=512):
    """Compute a feature vector for each beat boundary.
    Returns (n_beats, n_features) array.
    Features: chroma(12) + spectral_contrast(7) + onset_str(1) + rms(1) = 21."""
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, hop_length=hop)
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=audio, hop_length=hop)[0]

    beat_frames = librosa.time_to_frames(beats_sec, sr=sr, hop_length=hop)
    beat_frames = np.clip(beat_frames, 0, chroma.shape[1] - 1)

    # Aggregate features in a small window around each beat (±2 frames)
    feats = []
    for bf in beat_frames:
        lo = max(0, bf - 2)
        hi = min(chroma.shape[1], bf + 3)
        f = np.concatenate([
            chroma[:, lo:hi].mean(axis=1),        # 12
            contrast[:, lo:hi].mean(axis=1),       # 7
            [onset_env[lo:hi].mean()],             # 1
            [rms[lo:hi].mean()],                   # 1
        ])
        feats.append(f)
    feats = np.array(feats, dtype=np.float32)

    # Normalize each feature dimension to [0,1] for fair distance comparison
    mins = feats.min(axis=0, keepdims=True)
    maxs = feats.max(axis=0, keepdims=True)
    rng = maxs - mins + 1e-9
    feats = (feats - mins) / rng
    return feats


def _score_loop_quality(audio, sr):
    """Score a loop candidate's musical quality (0-1).
    Harmonic richness (0.4) + rhythmic interest (0.3) + consistency (0.3)."""
    if len(audio) < sr * 0.5:
        return 0.0

    # Harmonic richness: mean chroma peak magnitude (tonal content)
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=512)
    harmonic = float(chroma.max(axis=0).mean())  # 0-1 range naturally

    # Rhythmic interest: onset rate (sweet spot 2-8 per second)
    onsets = librosa.onset.onset_detect(y=audio, sr=sr, units='time')
    dur = len(audio) / sr
    rate = len(onsets) / dur if dur > 0 else 0
    if rate < 0.5:
        rhythm = 0.1  # too static
    elif rate <= 8.0:
        rhythm = min(1.0, rate / 5.0)  # sweet spot
    else:
        rhythm = max(0.2, 1.0 - (rate - 8.0) / 10.0)  # too noisy

    # Consistency: RMS coefficient of variation (low = steady, high = gaps)
    rms = librosa.feature.rms(y=audio, hop_length=512)[0]
    rms_mean = float(rms.mean()) + 1e-9
    rms_cv = float(rms.std()) / rms_mean
    consistency = max(0.0, 1.0 - rms_cv)  # CV of 0 = perfect, CV of 1+ = bad

    return harmonic * 0.4 + rhythm * 0.3 + consistency * 0.3


def _find_loop_pairs(beats_sec, beat_feats, target_bars, target_bpm,
                     bar_tolerance=0.15, top_n=10, raw_audio=None, sr=SR):
    """Find beat pairs whose gap = target_bars at target_bpm with minimal
    feature distance + musical quality. Returns list of (score, start_beat_idx, end_beat_idx)."""
    beat_dur = 60.0 / target_bpm
    target_dur = target_bars * 4 * beat_dur
    tol = target_dur * bar_tolerance

    n = len(beats_sec)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            gap = beats_sec[j] - beats_sec[i]
            if gap < target_dur - tol:
                continue
            if gap > target_dur + tol:
                break
            # L1 distance (seamlessness) — lower = better loop boundary
            l1_dist = float(np.sum(np.abs(beat_feats[i] - beat_feats[j])))

            # Musical quality score — higher = better content
            quality = 0.5  # default if no audio
            if raw_audio is not None:
                seg_start = int(beats_sec[i] * sr)
                seg_end = int(beats_sec[j] * sr)
                if seg_end <= len(raw_audio):
                    segment = raw_audio[seg_start:seg_end]
                    quality = _score_loop_quality(segment, sr)

            # Combined score: balance seamlessness with quality
            score = l1_dist * 0.5 + (1.0 - quality) * 3.0

            # Energy penalty for quiet segments
            seg_rms = beat_feats[i:j+1, -1].mean()
            if seg_rms < 0.1:
                score += 5.0
            pairs.append((score, i, j))

    pairs.sort(key=lambda x: x[0])
    return pairs[:top_n]


def score_loop_candidates(raw_sample, loop_period, sr=SR, top_n=5):
    """Score candidate positions by loop-ability + simplicity + energy.
    Returns list of (score, pos, loop_sim, rms, simplicity)."""
    loop_samp = int(loop_period * sr)
    chroma_hop = 512
    step = int(0.5 * sr)
    candidates = []
    for pos in range(0, len(raw_sample) - loop_samp, step):
        seg = raw_sample[pos:pos + loop_samp]
        rms = np.sqrt(np.mean(seg ** 2))
        if rms < 0.01:
            continue
        ch = librosa.feature.chroma_cqt(y=seg, sr=sr, hop_length=chroma_hop)
        n_compare = min(8, ch.shape[1] // 4)
        ch_start = ch[:, :n_compare].mean(axis=1)
        ch_end = ch[:, -n_compare:].mean(axis=1)
        loop_sim = float(np.dot(ch_start, ch_end) /
                         (np.linalg.norm(ch_start) * np.linalg.norm(ch_end) + 1e-9))
        ch_mean = ch.mean(axis=1)
        ch_norm = ch_mean / (ch_mean.sum() + 1e-9)
        entropy = -float(np.sum(ch_norm * np.log2(ch_norm + 1e-9)))
        simplicity = 1.0 - (entropy / np.log2(12))
        score = 0.45 * loop_sim + 0.25 * simplicity + 0.30 * (rms / 0.2)
        candidates.append((score, pos, loop_sim, rms, simplicity))
    candidates.sort(key=lambda x: -x[0])
    return candidates[:top_n]


def extract_bar_aligned_loop(chunk, sr=SR, target_dur=None, target_bpm=None):
    """Extract a loop snapped to beat boundaries using multi-feature scoring.
    Returns (loop_audio, n_bars, bar_times, loop_start_samp)."""
    beats_sec = _get_beats(chunk, sr, target_bpm=target_bpm)

    if len(beats_sec) < 4:
        # Not enough beats — fallback to raw cut
        loop_samp = int(target_dur * sr) if target_dur else len(chunk)
        loop = chunk[:min(loop_samp, len(chunk))].copy()
        n_bars = max(1, round((len(loop) / sr) / (target_dur or 4.0)))
        return loop, n_bars, np.linspace(0, len(loop)/sr, n_bars+1), 0

    # Estimate BPM from beat intervals if not provided
    if target_bpm is None:
        intervals = np.diff(beats_sec)
        intervals = intervals[(intervals > 0.15) & (intervals < 2.0)]
        if len(intervals) >= 2:
            target_bpm = 60.0 / np.median(intervals)
        else:
            target_bpm = 120.0

    beat_dur = 60.0 / target_bpm

    # Determine target bar count from target_dur
    if target_dur:
        target_bars = max(1, round(target_dur / (4 * beat_dur)))
    else:
        target_bars = 4

    # Compute multi-feature vectors at each beat
    beat_feats = _beat_features(chunk, sr, beats_sec)

    # Find best loop pairs
    pairs = _find_loop_pairs(beats_sec, beat_feats, target_bars, target_bpm)

    if not pairs:
        # Relax tolerance and try again
        pairs = _find_loop_pairs(beats_sec, beat_feats, target_bars, target_bpm,
                                  bar_tolerance=0.30)

    if pairs:
        best_dist, bi, bj = pairs[0]
        loop_start = int(beats_sec[bi] * sr)
        loop_end = int(beats_sec[bj] * sr)
        print(f'    Best loop pair: beat {bi}->{bj}  dist={best_dist:.3f}  '
              f'dur={beats_sec[bj]-beats_sec[bi]:.2f}s')
    else:
        # Absolute fallback: first N bars from first beat
        loop_start = int(beats_sec[0] * sr)
        loop_dur_samp = int(target_bars * 4 * beat_dur * sr)
        loop_end = min(loop_start + loop_dur_samp, len(chunk))

    loop = chunk[loop_start:loop_end].copy()
    actual_dur = len(loop) / sr

    # Compute n_bars from actual duration and target BPM
    n_bars = max(1, round(actual_dur / (4 * beat_dur)))

    # Short crossfade for seamless looping
    xf = min(int(0.005 * sr), len(loop) // 8)
    if xf > 0 and len(loop) > xf * 2:
        loop[:xf] *= np.linspace(0, 1, xf).astype(np.float32)
        loop[-xf:] *= np.linspace(1, 0, xf).astype(np.float32)

    bar_times = np.linspace(0, actual_dur, n_bars + 1)
    return loop, n_bars, bar_times, loop_start


def extract_loop_at(raw_sample, loop_start_s, sr=SR, loop_end_s=None):
    """Extract a loop starting at a specific time. Uses local beat tracking."""
    chunk_start = int(loop_start_s * sr)
    chunk_end = min(chunk_start + int(30.0 * sr), len(raw_sample))
    chunk = raw_sample[chunk_start:chunk_end].copy()

    if loop_end_s is not None:
        target_dur = loop_end_s - loop_start_s
    else:
        target_dur = detect_loop_period(chunk, sr)
        print(f'  Auto-detected loop period: {target_dur:.2f}s')

    loop, n_bars, bar_times, loop_start_samp = extract_bar_aligned_loop(
        chunk, sr, target_dur)
    abs_start = loop_start_s + loop_start_samp / sr
    print(f'  Loop: {abs_start:.2f}s ({len(loop)/sr:.2f}s, {n_bars} bars)')
    print(f'  Bar boundaries: {bar_times}')
    return loop, n_bars


def extract_loop_auto(raw_sample, sr=SR, target_bpm=None):
    """Auto-detect best loop using beat-aligned multi-feature scoring.

    Algorithm (inspired by librosa_loopfinder):
      1. Beat-track the full sample (madmom primary, librosa fallback)
      2. Compute feature vectors at each beat: chroma(12) + spectral_contrast(7)
         + onset_strength(1) + RMS(1)
      3. Normalize features, compute pairwise L1 distance between all beat pairs
      4. Filter pairs by target loop duration (2 or 4 bars at detected BPM)
      5. Rank by distance (lower = more seamless loop) + energy penalty
      6. Extract the winning segment with crossfade
    """
    print('  Beat-aligned loop detection ...')

    # Step 1: Get beats for the whole sample
    beats_sec = _get_beats(raw_sample, sr, target_bpm=target_bpm)
    if len(beats_sec) < 4:
        print('    Too few beats, falling back to chroma period detection')
        loop_period = detect_loop_period(raw_sample, sr)
        candidates = score_loop_candidates(raw_sample, loop_period, sr)
        best_pos = candidates[0][1]
        loop = raw_sample[best_pos:best_pos + int(loop_period * sr)].copy()
        n_bars = max(1, round(loop_period / 2.0))
        return loop, n_bars

    # Estimate BPM from beats
    intervals = np.diff(beats_sec)
    intervals = intervals[(intervals > 0.15) & (intervals < 2.0)]
    detected_bpm = 60.0 / np.median(intervals) if len(intervals) >= 2 else 120.0

    # Half/double toward target if provided
    if target_bpm:
        best_bpm, best_dist = detected_bpm, abs(detected_bpm - target_bpm)
        for mult in [0.5, 2.0]:
            adj = detected_bpm * mult
            d = abs(adj - target_bpm)
            if d < best_dist:
                best_bpm, best_dist = adj, d
        detected_bpm = best_bpm
    print(f'    BPM from beats: {detected_bpm:.1f}')

    beat_dur = 60.0 / detected_bpm

    # Step 2: Compute features at each beat
    beat_feats = _beat_features(raw_sample, sr, beats_sec)
    print(f'    Feature matrix: {beat_feats.shape[0]} beats x {beat_feats.shape[1]} features')

    # Step 3-5: Try 4-bar loop first, then 2-bar
    best_pairs = None
    best_n_bars = 4
    for try_bars in [4, 2, 8]:
        pairs = _find_loop_pairs(beats_sec, beat_feats, try_bars, detected_bpm,
                                  bar_tolerance=0.15, raw_audio=raw_sample, sr=sr)
        if pairs:
            print(f'    {try_bars}-bar candidates: {len(pairs)} '
                  f'(best dist={pairs[0][0]:.3f})')
            if best_pairs is None or pairs[0][0] < best_pairs[0][0]:
                best_pairs = pairs
                best_n_bars = try_bars

    if not best_pairs:
        # Relax tolerance
        for try_bars in [4, 2]:
            pairs = _find_loop_pairs(beats_sec, beat_feats, try_bars, detected_bpm,
                                      bar_tolerance=0.30, raw_audio=raw_sample, sr=sr)
            if pairs:
                best_pairs = pairs
                best_n_bars = try_bars
                break

    if not best_pairs:
        print('    No beat-aligned pairs found, falling back to chroma period')
        loop_period = detect_loop_period(raw_sample, sr)
        candidates = score_loop_candidates(raw_sample, loop_period, sr)
        best_pos = candidates[0][1]
        loop = raw_sample[best_pos:best_pos + int(loop_period * sr)].copy()
        n_bars = max(1, round(loop_period / (4 * beat_dur)))
        return loop, n_bars

    # Step 6: Extract winning segment
    dist, bi, bj = best_pairs[0]
    loop_start = int(beats_sec[bi] * sr)
    loop_end = int(beats_sec[bj] * sr)
    loop = raw_sample[loop_start:loop_end].copy()
    actual_dur = len(loop) / sr
    n_bars = max(1, round(actual_dur / (4 * beat_dur)))

    # Print top 3 candidates
    for rank, (d, i, j) in enumerate(best_pairs[:3]):
        t = beats_sec[i]
        dur = beats_sec[j] - beats_sec[i]
        mins, secs = int(t // 60), t % 60
        seg = raw_sample[int(beats_sec[i]*sr):int(beats_sec[j]*sr)]
        rms = np.sqrt(np.mean(seg ** 2))
        print(f'  #{rank+1}: {mins}:{secs:05.2f} (dist={d:.3f}, dur={dur:.2f}s, '
              f'rms={rms:.4f}, bars={round(dur/(4*beat_dur))})')

    # Crossfade
    xf = min(int(0.005 * sr), len(loop) // 8)
    if xf > 0 and len(loop) > xf * 2:
        loop[:xf] *= np.linspace(0, 1, xf).astype(np.float32)
        loop[-xf:] *= np.linspace(1, 0, xf).astype(np.float32)

    print(f'  Selected: {beats_sec[bi]:.2f}s ({actual_dur:.2f}s, {n_bars} bars, '
          f'{detected_bpm:.1f} BPM)')
    return loop, n_bars


def detect_and_align_loop(loop, sr=SR, bpm_range=(70, 200), bpm_hint=None, vinyl_mode=False):
    """Detect loop BPM, compute bar count, and align/stretch to clean bars.

    Consolidated function that handles all edge cases:
    - Beat tracking failure → fallback to bpm_hint or range midpoint
    - BPM outside genre range → stretch to hint (vinyl_mode uses resample)
    - Non-integer bar count → time-stretch to nearest integer bars
    - Zero-division guards

    Args:
        loop: mono audio array
        sr: sample rate
        bpm_range: (lo, hi) tuple for genre
        bpm_hint: target BPM (from --bpm or genre default)
        vinyl_mode: if True, uses resampling (pitch+speed linked) instead of
                    phase vocoder time-stretch

    Returns: (aligned_loop, bpm, n_bars)
    """
    from scipy.signal import resample as scipy_resample

    lo, hi = bpm_range
    loop_dur = len(loop) / sr

    # Step 1: Beat-track the loop
    tempo_arr = librosa.beat.beat_track(y=loop, sr=sr)
    loop_bpm_raw = float(np.atleast_1d(tempo_arr[0])[0]) if hasattr(tempo_arr[0], '__len__') else float(tempo_arr[0])

    # Step 2: Fallback if beat tracking fails
    if loop_bpm_raw < 1:
        loop_bpm_raw = bpm_hint if bpm_hint else (lo + hi) / 2
        print(f'  Beat tracking failed, using {loop_bpm_raw:.0f} BPM fallback')

    # Step 3: Adjust to genre range (1x, 0.5x, 2x)
    loop_bpm = loop_bpm_raw
    in_range = False
    for mult in [1.0, 0.5, 2.0]:
        adj = loop_bpm_raw * mult
        if lo <= adj <= hi:
            loop_bpm = adj
            in_range = True
            break

    # Step 4: Compute bar count
    bar_dur = 60.0 / loop_bpm * 4
    if bar_dur < 0.1:
        bar_dur = 60.0 / ((lo + hi) / 2) * 4
    n_bars = max(1, round(loop_dur / bar_dur))

    # Step 5/6: Align to clean bars
    if in_range:
        # In genre range — time-stretch to align cleanly to N bars
        target_dur = n_bars * bar_dur
        stretch_ratio = abs(loop_dur - target_dur) / loop_dur if loop_dur > 0 else 0
        if stretch_ratio > 0.01:
            if vinyl_mode:
                new_len = int(len(loop) * (loop_dur / target_dur))
                loop = scipy_resample(loop, new_len).astype(np.float32)
                print(f'  Vinyl-aligned: {loop_dur:.2f}s → {target_dur:.2f}s ({n_bars} bars at {loop_bpm:.1f} BPM)')
            else:
                loop = librosa.effects.time_stretch(loop, rate=loop_dur / target_dur).astype(np.float32)
                print(f'  Time-aligned: {loop_dur:.2f}s → {target_dur:.2f}s ({n_bars} bars at {loop_bpm:.1f} BPM)')
            loop_dur = len(loop) / sr
        bpm = loop_bpm
    else:
        # Out of genre range — stretch to bpm_hint
        target_bpm = bpm_hint if bpm_hint else (lo + hi) / 2
        target_bar_dur = 60.0 / target_bpm * 4
        target_dur = n_bars * target_bar_dur
        if loop_dur > 0 and target_dur > 0:
            if vinyl_mode:
                # Resample = pitch + speed linked (vinyl slowdown)
                new_len = int(len(loop) * (target_dur / loop_dur))
                loop = scipy_resample(loop, new_len).astype(np.float32)
                print(f'  Vinyl-stretched: {loop_bpm:.1f} → {target_bpm:.1f} BPM ({n_bars} bars, pitch shifted)')
            else:
                loop = librosa.effects.time_stretch(loop, rate=loop_dur / target_dur).astype(np.float32)
                print(f'  Time-stretched: {loop_bpm:.1f} → {target_bpm:.1f} BPM ({n_bars} bars)')
            loop_dur = len(loop) / sr
        bpm = target_bpm

    # Recompute bar count from final loop
    final_bar_dur = 60.0 / bpm * 4
    n_bars = max(1, round(loop_dur / final_bar_dur)) if final_bar_dur > 0 else 1

    print(f'  Loop BPM (beat-tracked): {loop_bpm_raw:.1f} | Final: {bpm:.1f} BPM, {n_bars} bars')
    return loop, bpm, n_bars
