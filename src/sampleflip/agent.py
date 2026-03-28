"""LLM agent — plans beats, generates drums, picks samples."""

import json
import anthropic


def _call(system, user_msg, max_tokens=1024):
    """Single Claude API call. Returns parsed JSON."""
    client = anthropic.Anthropic()
    response = client.messages.create(
        model='claude-sonnet-4-20250514',
        max_tokens=max_tokens,
        system=system,
        messages=[{'role': 'user', 'content': user_msg}],
    )
    raw = response.content[0].text
    if '```' in raw:
        raw = raw.split('```json')[-1] if '```json' in raw else raw.split('```')[-2]
        raw = raw.replace('```', '').strip()
    return json.loads(raw)


# ── 1. Plan beat (genre, BPM, search query, name) ──

PLAN_SYSTEM = """You are a beat production assistant. Given a user's description, decide:

1. genre — one of: trap, boombap, jazzhouse, progressive_house, rnb, drill, melodic_trap, 2hollis, techno, breakcore
2. bpm — appropriate tempo:
   trap: 140-160, boombap: 75-95, jazzhouse: 120-128, progressive_house: 126-132,
   rnb: 65-85, drill: 138-144, melodic_trap: 135-150, 2hollis: 140-160, techno: 128-140, breakcore: 160-180
3. search_query — YouTube query to find a sample/loop (NOT tutorials or full songs). Include "loop", "sample pack", "free". Be specific about instruments.
4. name — short beat name (2-3 words, underscores)

Return ONLY JSON: {"genre": "...", "bpm": 140, "search_query": "...", "name": "..."}"""


def plan_beat(prompt, genre_override=None, bpm_override=None):
    """Parse user intent into beat parameters."""
    user_msg = prompt
    if genre_override:
        user_msg += f"\n\nNote: genre must be {genre_override}"
    if bpm_override:
        user_msg += f"\n\nNote: BPM must be {bpm_override}"

    result = _call(PLAN_SYSTEM, user_msg, max_tokens=256)

    if genre_override:
        result['genre'] = genre_override
    if bpm_override:
        result['bpm'] = bpm_override
    return result


# ── 2. Pick best search result ──

PICK_SYSTEM = """You are choosing the best YouTube result for sampling into a beat.

Pick the result most likely to be an ACTUAL sample pack, loop kit, or melody loop.

PREFER:
- Titles with "loop kit", "sample pack", "free", "melody", "loops"
- Short duration (under 5 min = likely a preview of loops)
- Channels that are loop/sample producers

AVOID:
- Tutorials ("how to make", "tutorial", "FL Studio")
- Full songs or albums (over 10 min)
- Beat showcases ("type beat")
- Reaction videos, reviews

Return ONLY JSON: {"pick": <1-based index>, "reason": "short explanation"}"""


def pick_best_result(results, genre, prompt):
    """LLM picks the best YouTube result for sampling."""
    listings = []
    for i, r in enumerate(results, 1):
        listings.append(f'{i}. "{r["title"]}" ({r["duration_str"]}) — {r["channel"]}')

    user_msg = f'Genre: {genre}\nUser wants: {prompt}\n\nResults:\n' + '\n'.join(listings)
    result = _call(PICK_SYSTEM, user_msg, max_tokens=128)
    idx = int(result['pick']) - 1
    return max(0, min(idx, len(results) - 1))


# ── 3. Generate drum pattern ──

DRUMS_SYSTEM = """You are an expert drum programmer. Generate a drum pattern as MIDI events.

RULES:
- Positions are BEATS in a 4/4 bar (0.0 to 3.75)
- Beat 0=the "1", Beat 1=the "2", Beat 2=the "3", Beat 3=the "4"
- Subdivisions: 0.5=8th, 0.25=16th, 0.125=32nd
- Each event: [beat_position, gm_drum_note, velocity]
- Velocity: ghost=30-40, normal=70-90, accent=95-110

GM DRUM NOTES:
  36=Kick  38=Snare  39=Clap  42=Closed HH  46=Open HH  49=Crash  51=Ride  37=Rim

OUTPUT FORMAT:
{
  "patterns": {
    "A": [[beat, note, vel], ...],
    "B": [[beat, note, vel], ...],
    "fill": [[beat, note, vel], ...],
    "silent": []
  },
  "bar_sequence": ["pattern_id", ...]
}

REQUIREMENTS:
- 2-4 main patterns (A, B, C) with subtle differences
- 1-2 fill patterns for transitions
- "silent" pattern (empty) for intros/outros
- bar_sequence length MUST equal total bars requested
- Use "silent" for intro/outro bars, fills before section changes
- Match the genre and vibe described"""


def generate_drums(prompt, genre, bpm, nbars, arrangement):
    """LLM generates a custom drum pattern for this specific beat."""
    arr_lines = []
    for sec in arrangement:
        name, start, end, kick_on, fx = sec
        drums = 'full drums' if kick_on else 'no drums'
        arr_lines.append(f'  Bars {start}-{end-1}: {name} ({drums})')

    user_msg = f"""Generate drums for: {prompt}
Genre: {genre} | BPM: {bpm} | Total bars: {nbars}

Arrangement:
{chr(10).join(arr_lines)}

Place fills in the last bar before each section change. Use "silent" for "no drums" sections."""

    result = _call(DRUMS_SYSTEM, user_msg, max_tokens=8192)

    # Validate
    patterns = result.get('patterns', {})
    bar_sequence = result.get('bar_sequence', [])

    if 'silent' not in patterns:
        patterns['silent'] = []

    # Pad/trim bar_sequence
    if len(bar_sequence) < nbars:
        last = bar_sequence[-1] if bar_sequence else list(patterns.keys())[0]
        bar_sequence.extend([last] * (nbars - len(bar_sequence)))
    bar_sequence = bar_sequence[:nbars]

    # Resolve unknown refs
    valid_ids = set(patterns.keys())
    first_id = [k for k in patterns if k != 'silent'][0] if len(patterns) > 1 else 'silent'
    for i, pid in enumerate(bar_sequence):
        if pid not in valid_ids:
            bar_sequence[i] = first_id

    # Clamp values
    for pat_id in patterns:
        clean = []
        for ev in patterns[pat_id]:
            if isinstance(ev, (list, tuple)) and len(ev) >= 3:
                beat = max(0.0, min(3.875, float(ev[0])))
                note = int(ev[1])
                vel = max(1, min(127, int(ev[2])))
                if note in {36, 37, 38, 39, 42, 46, 49, 51, 56}:
                    clean.append([beat, note, vel])
        patterns[pat_id] = clean

    return {'patterns': patterns, 'bar_sequence': bar_sequence}


# ── 4. Generate bass pattern ──

BASS_SYSTEM = """You are a bass programmer. Given a genre, chord progression, and vibe, decide the bass pattern type.

Available patterns:
- root_only: one bass note per bar on beat 1 (simplest)
- root_octave: root on beat 1 + octave on beat 3 (boom-bap bounce)
- four_pulse: root on every beat (house four-on-the-floor)
- bounce: root on beat 1 + 5th on beat 2.5 (trap bounce)
- walking: root → 3rd → 5th → octave across 4 beats (R&B/jazz walk)
- drill_slide: root on beat 1, root on beat 2, 5th on beat 3 (drill)

Return ONLY JSON: {"bass_pattern": "..."}"""


def pick_bass_pattern(prompt, genre, chords_description=None):
    """LLM picks the best bass pattern for this beat."""
    user_msg = f'Genre: {genre}\nVibe: {prompt}'
    if chords_description:
        user_msg += f'\nChords: {chords_description}'

    result = _call(BASS_SYSTEM, user_msg, max_tokens=64)
    valid = ['root_only', 'root_octave', 'four_pulse', 'bounce', 'walking', 'drill_slide']
    pat = result.get('bass_pattern', 'root_only')
    return pat if pat in valid else 'root_only'
