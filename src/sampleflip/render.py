"""Render wrapper — calls core pipeline with portable paths."""

import os
import sys


def render_beat(sample_path, name, genre='trap', bpm=None, bars=None,
                loop_start=None, loop_end=None, drums_json=None,
                vinyl_slow=False, no_bass=False, lofi=None,
                output_dir=None, kit_dir=None):
    """Generate a beat from a sample file.

    Args:
        sample_path: path to WAV/MP3 sample
        name: beat name (used for output filename)
        genre: one of the supported genres
        bpm: override BPM (auto-detected if None)
        bars: override bar count
        loop_start: manual loop start in seconds
        loop_end: manual loop end in seconds
        drums_json: path to drum pattern JSON
        vinyl_slow: pitch+speed linked slowdown
        no_bass: disable bass
        lofi: lo-fi intensity 0-1
        output_dir: output directory (default: ./output)
        kit_dir: path to drum kit samples directory

    Returns: path to output MP3
    """
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Add core to path so render_beat.py can find sibling modules
    core_dir = os.path.join(os.path.dirname(__file__), 'core')
    if core_dir not in sys.path:
        sys.path.insert(0, core_dir)

    # Set kit directory via environment variable (core reads this)
    if kit_dir:
        os.environ['SAMPLEFLIP_KIT_DIR'] = kit_dir

    # Override output directory
    os.environ['SAMPLEFLIP_OUTPUT_DIR'] = output_dir

    from sampleflip.core.render_beat import render

    result = render(
        sample_path, name, genre=genre,
        bpm_hint=bpm, nbars=bars,
        loop_start=loop_start, loop_end=loop_end,
        lofi=lofi, no_bass=no_bass,
        vinyl_slow=vinyl_slow, drums_json=drums_json,
    )
    return result
