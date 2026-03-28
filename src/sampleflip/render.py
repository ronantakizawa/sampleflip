"""Render wrapper — calls core pipeline with portable paths."""

import os
import sys


def render_beat(sample_path, name, genre='trap', bpm=None, bars=None,
                loop_start=None, loop_end=None, drums_json=None,
                vinyl_slow=False, no_bass=False, lofi=None,
                output_dir=None, kit_dir=None, bass_pattern_type=None):
    """Generate a beat from a sample file."""
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'output')
    os.makedirs(output_dir, exist_ok=True)

    core_dir = os.path.join(os.path.dirname(__file__), 'core')
    if core_dir not in sys.path:
        sys.path.insert(0, core_dir)

    if kit_dir:
        os.environ['SAMPLEFLIP_KIT_DIR'] = kit_dir
    os.environ['SAMPLEFLIP_OUTPUT_DIR'] = output_dir

    # Override bass pattern type if LLM picked one
    if bass_pattern_type:
        os.environ['SAMPLEFLIP_BASS_PATTERN'] = bass_pattern_type

    from sampleflip.core.render_beat import render

    result = render(
        sample_path, name, genre=genre,
        bpm_hint=bpm, nbars=bars,
        loop_start=loop_start, loop_end=loop_end,
        lofi=lofi, no_bass=no_bass,
        vinyl_slow=vinyl_slow, drums_json=drums_json,
    )
    return result
