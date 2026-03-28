"""LLM agent — parses user prompt into beat generation parameters."""

import json
import anthropic

SYSTEM_PROMPT = """You are a beat production assistant. Given a user's description of the beat they want, decide:

1. genre — one of: trap, boombap, jazzhouse, progressive_house, rnb, drill, melodic_trap, 2hollis, techno, breakcore
2. bpm — appropriate tempo for the genre:
   - trap: 140-160
   - boombap: 75-95
   - jazzhouse: 120-128
   - progressive_house: 126-132
   - rnb: 65-85
   - drill: 138-144
   - melodic_trap: 135-150
   - 2hollis: 140-160
   - techno: 128-140
   - breakcore: 160-180
3. search_query — a YouTube search query that will find a good sample/loop for this beat. The query should:
   - Target actual sample packs, loop kits, or melody loops (NOT tutorials or full songs)
   - Include words like "loop", "sample pack", "free", "melody"
   - Match the mood/vibe described by the user
   - Be specific about instruments (piano, guitar, strings, vocal, etc.)
4. name — a short beat name (2-3 words, no spaces, use underscores)

Return ONLY valid JSON, no explanation:
{"genre": "...", "bpm": 140, "search_query": "...", "name": "..."}"""


def plan_beat(prompt, genre_override=None, bpm_override=None):
    """Use Claude to parse user intent into beat parameters.

    Args:
        prompt: natural language description of the beat
        genre_override: force a specific genre (LLM still picks query/name)
        bpm_override: force a specific BPM

    Returns: dict with keys: genre, bpm, search_query, name
    """
    client = anthropic.Anthropic()

    user_msg = prompt
    if genre_override:
        user_msg += f"\n\nNote: genre must be {genre_override}"
    if bpm_override:
        user_msg += f"\n\nNote: BPM must be {bpm_override}"

    response = client.messages.create(
        model='claude-sonnet-4-20250514',
        max_tokens=256,
        system=SYSTEM_PROMPT,
        messages=[{'role': 'user', 'content': user_msg}],
    )

    raw = response.content[0].text
    if '```' in raw:
        raw = raw.split('```json')[-1] if '```json' in raw else raw.split('```')[-2]
        raw = raw.replace('```', '').strip()

    result = json.loads(raw)

    # Apply overrides
    if genre_override:
        result['genre'] = genre_override
    if bpm_override:
        result['bpm'] = bpm_override

    return result
