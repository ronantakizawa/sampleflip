"""SampleFlip CLI — Describe a beat, get a beat."""

import json
import os
import tempfile
import click

GENRES = [
    'trap', 'boombap', 'jazzhouse', 'progressive_house', 'rnb',
    'drill', 'melodic_trap', '2hollis', 'techno', 'breakcore',
]


@click.command()
@click.argument('prompt')
@click.option('--genre', type=click.Choice(GENRES), default=None, help='Override genre')
@click.option('--bpm', type=float, default=None, help='Override BPM')
@click.option('--bars', type=int, default=None, help='Number of bars')
@click.option('--loop-start', type=float, default=None, help='Loop start (seconds)')
@click.option('--loop-end', type=float, default=None, help='Loop end (seconds)')
@click.option('--vinyl-slow', is_flag=True, help='Pitch+speed linked slowdown')
@click.option('--no-bass', is_flag=True, help='Disable bass')
@click.option('--lofi', type=float, default=None, help='Lo-fi intensity (0-1)')
@click.option('--output', type=click.Path(), default=None, help='Output directory')
@click.option('--kit', type=click.Path(exists=True), default=None, help='Drum kit samples directory')
@click.version_option()
def main(prompt, genre, bpm, bars, loop_start, loop_end,
         vinyl_slow, no_bass, lofi, output, kit):
    """Describe a beat, get a beat.

    Examples:

      sampleflip "dark trap beat with piano"

      sampleflip "jazzy house with sax"

      sampleflip "aggressive drill beat" --bpm 142
    """
    from sampleflip.agent import plan_beat, plan_all
    from sampleflip.search import search_youtube
    from sampleflip.download import download_youtube
    from sampleflip.render import render_beat

    # Step 1: Quick LLM call to get search query
    click.echo(f'\nPlanning beat...')
    try:
        quick_plan = plan_beat(prompt, genre_override=genre, bpm_override=bpm)
    except Exception as e:
        click.echo(f'Error: {e}', err=True)
        click.echo('Make sure ANTHROPIC_API_KEY is set.', err=True)
        raise SystemExit(1)

    g = quick_plan['genre']
    q = quick_plan['search_query']
    click.echo(f'  Genre: {g} | Query: "{q}"')

    # Step 2: Search YouTube
    click.echo(f'\nSearching YouTube...')
    try:
        results = search_youtube(q, count=5)
    except RuntimeError as e:
        click.echo(f'Search failed: {e}', err=True)
        raise SystemExit(1)

    if not results:
        click.echo('No results found. Try a different description.', err=True)
        raise SystemExit(1)

    click.echo(f'  Found {len(results)} results')

    # Step 3: Batched LLM call — pick result + drums + bass in ONE call
    import sys
    core_dir = os.path.join(os.path.dirname(__file__), 'core')
    if core_dir not in sys.path:
        sys.path.insert(0, core_dir)
    from sampleflip.core.render_beat import GENRE_CONFIGS
    cfg = GENRE_CONFIGS[g]
    nbars_actual = bars or cfg['bars']
    arrangement = cfg['arrangement']

    click.echo(f'Generating drums + picking sample (single LLM call)...')
    try:
        batch = plan_all(prompt, results, nbars_actual, arrangement,
                         genre_override=genre, bpm_override=bpm)
        b = batch['plan']['bpm']
        name = batch['plan']['name']
        best_idx = batch['pick']
        bass_pat = batch['bass_pattern']
        drum_data = batch['drums']

        n_pats = len([k for k in drum_data['patterns'] if k != 'silent'])
        click.echo(f'  BPM: {b} | Name: {name}')
        click.echo(f'  Bass: {bass_pat} | Drums: {n_pats} patterns')
    except Exception as e:
        click.echo(f'  Batched call failed ({e}), falling back to individual calls')
        from sampleflip.agent import pick_best_result, generate_drums, pick_bass_pattern
        b = quick_plan['bpm']
        name = quick_plan['name']
        try:
            best_idx = pick_best_result(results, g, prompt)
        except Exception:
            best_idx = 0
        try:
            bass_pat = pick_bass_pattern(prompt, g)
        except Exception:
            bass_pat = None
        try:
            drum_data = generate_drums(prompt, g, b, nbars_actual, arrangement)
        except Exception:
            drum_data = None

    # Step 4: Download + drum check (try next result if drums detected)
    from sampleflip.drum_check import has_drums as check_drums

    sample_path = None
    tried = set()
    candidates = [best_idx] + [i for i in range(len(results)) if i != best_idx]

    for idx in candidates:
        if idx in tried:
            continue
        tried.add(idx)
        selected = results[idx]
        click.echo(f'\n  Trying: {selected["title"]} ({selected["duration_str"]})')

        try:
            sample_path, size_mb = download_youtube(selected['url'])
            click.echo(f'  Downloaded ({size_mb:.1f} MB)')
        except RuntimeError as e:
            click.echo(f'  Download failed: {e}')
            continue

        drums_found, drum_score, drum_details = check_drums(sample_path)
        if drums_found:
            click.echo(f'  Drums detected (score={drum_score:.2f}: {drum_details}) — skipping')
            sample_path = None
            continue
        else:
            click.echo(f'  Clean sample (score={drum_score:.2f}) — using it')
            break

    if sample_path is None:
        click.echo('All results contain drums. Using last downloaded anyway.', err=True)
        selected = results[best_idx]
        sample_path, size_mb = download_youtube(selected['url'])

    # Step 5: Write drum JSON
    drums_json = None
    if drum_data:
        drums_json = os.path.join(tempfile.gettempdir(), f'sampleflip_drums_{name}.json')
        with open(drums_json, 'w') as f:
            json.dump(drum_data, f)

    # Step 6: Set bass pattern
    if bass_pat:
        os.environ['SAMPLEFLIP_BASS_PATTERN'] = bass_pat

    # Step 7: Generate beat
    click.echo(f'\nGenerating {g} beat: "{name}"...\n')
    try:
        render_beat(
            sample_path, name, genre=g, bpm=b, bars=bars,
            loop_start=loop_start, loop_end=loop_end,
            drums_json=drums_json, vinyl_slow=vinyl_slow,
            no_bass=no_bass, lofi=lofi,
            output_dir=output, kit_dir=kit,
            bass_pattern_type=bass_pat,
        )
    except Exception as e:
        click.echo(f'\nRender failed: {e}', err=True)
        raise SystemExit(1)

    click.echo(f'\nDone!')


if __name__ == '__main__':
    main()
