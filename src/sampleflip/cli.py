"""SampleFlip CLI — Describe a beat, get a beat."""

import os
import re
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
@click.option('--drums', type=click.Path(exists=True), default=None, help='Drum pattern JSON')
@click.option('--loop-start', type=float, default=None, help='Loop start (seconds)')
@click.option('--loop-end', type=float, default=None, help='Loop end (seconds)')
@click.option('--vinyl-slow', is_flag=True, help='Pitch+speed linked slowdown')
@click.option('--no-bass', is_flag=True, help='Disable bass')
@click.option('--lofi', type=float, default=None, help='Lo-fi intensity (0-1)')
@click.option('--output', type=click.Path(), default=None, help='Output directory')
@click.option('--kit', type=click.Path(exists=True), default=None, help='Drum kit samples directory')
@click.version_option()
def main(prompt, genre, bpm, bars, drums, loop_start, loop_end,
         vinyl_slow, no_bass, lofi, output, kit):
    """Describe a beat, get a beat.

    Examples:

      sampleflip "dark trap beat with piano"

      sampleflip "jazzy house with sax"

      sampleflip "aggressive drill beat" --bpm 142
    """
    from sampleflip.agent import plan_beat
    from sampleflip.search import search_youtube
    from sampleflip.download import download_youtube
    from sampleflip.render import render_beat

    # Step 1: LLM plans the beat
    click.echo(f'\nPlanning beat...')
    try:
        plan = plan_beat(prompt, genre_override=genre, bpm_override=bpm)
    except Exception as e:
        click.echo(f'Error: {e}', err=True)
        click.echo('Make sure ANTHROPIC_API_KEY is set.', err=True)
        raise SystemExit(1)

    g = plan['genre']
    b = plan['bpm']
    q = plan['search_query']
    name = plan['name']

    click.echo(f'  Genre: {g} | BPM: {b} | Name: {name}')
    click.echo(f'  Query: "{q}"')

    # Step 2: Search YouTube
    click.echo(f'\nSearching YouTube...')
    try:
        results = search_youtube(q, count=3)
    except RuntimeError as e:
        click.echo(f'Search failed: {e}', err=True)
        raise SystemExit(1)

    if not results:
        click.echo('No results found. Try a different description.', err=True)
        raise SystemExit(1)

    # Pick first result
    selected = results[0]
    click.echo(f'  Found: {selected["title"]} ({selected["duration_str"]})')

    # Step 3: Download
    click.echo(f'\nDownloading...')
    try:
        sample_path, size_mb = download_youtube(selected['url'])
        click.echo(f'  Done ({size_mb:.1f} MB)')
    except RuntimeError as e:
        click.echo(f'Download failed: {e}', err=True)
        raise SystemExit(1)

    # Step 4: Generate beat
    click.echo(f'\nGenerating {g} beat: "{name}"...\n')
    try:
        render_beat(
            sample_path, name, genre=g, bpm=b, bars=bars,
            loop_start=loop_start, loop_end=loop_end,
            drums_json=drums, vinyl_slow=vinyl_slow,
            no_bass=no_bass, lofi=lofi,
            output_dir=output, kit_dir=kit,
        )
    except Exception as e:
        click.echo(f'\nRender failed: {e}', err=True)
        raise SystemExit(1)

    click.echo(f'\nDone!')


if __name__ == '__main__':
    main()
