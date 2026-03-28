"""SampleFlip CLI — Search YouTube, generate beats."""

import os
import re
import click

GENRES = {
    'trap': 'Dark 808s, rolling hats, half-time clap (140-165 BPM)',
    'boombap': 'Vinyl chops, drum breaks, lo-fi feel (70-100 BPM)',
    'jazzhouse': 'Four-on-the-floor, ride, warm pads (115-135 BPM)',
    'progressive_house': 'Big drops, filter sweeps, saw leads (118-135 BPM)',
    'rnb': 'Soft drums, jazz chords, swing (60-95 BPM)',
    'drill': '808 slides, dark piano, aggressive hats (130-150 BPM)',
    'melodic_trap': 'Emotional melodies, bouncy 808, simpler hats (130-150 BPM)',
    '2hollis': 'Hyperpop, distorted, reese bass, chaotic (130-165 BPM)',
    'techno': 'Four-on-floor, acid, mechanical hats (124-145 BPM)',
    'breakcore': 'Chopped amen breaks, reese bass, 170 BPM',
}


@click.group()
@click.version_option()
def main():
    """SampleFlip — Search YouTube for samples, generate full beats."""
    pass


@main.command()
@click.argument('query')
@click.option('--count', default=5, help='Number of results (default: 5)')
def search(query, count):
    """Search YouTube for samples."""
    from sampleflip.search import search_youtube

    click.echo(f'\nSearching: "{query}"\n')
    try:
        results = search_youtube(query, count=count)
    except RuntimeError as e:
        click.echo(f'Error: {e}', err=True)
        raise SystemExit(1)

    if not results:
        click.echo('No results found.')
        return

    for i, r in enumerate(results, 1):
        click.echo(f'  {i}. {r["title"]} ({r["duration_str"]}) — {r["channel"]}')

    click.echo(f'\nUse: sampleflip generate <number> --genre <genre> to make a beat')


@main.command()
@click.argument('result_number', type=int)
@click.option('--genre', required=True, type=click.Choice(list(GENRES.keys())), help='Beat genre')
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
@click.option('--name', default=None, help='Beat name (default: derived from sample title)')
def generate(result_number, genre, bpm, bars, drums, loop_start, loop_end,
             vinyl_slow, no_bass, lofi, output, kit, name):
    """Download a search result and generate a beat from it."""
    from sampleflip.search import get_cached_results
    from sampleflip.download import download_youtube
    from sampleflip.render import render_beat

    # Load cached search results
    cache = get_cached_results()
    if cache is None:
        click.echo('No search results cached. Run "sampleflip search" first.', err=True)
        raise SystemExit(1)

    results = cache['results']
    if result_number < 1 or result_number > len(results):
        click.echo(f'Invalid result number. Choose 1-{len(results)}.', err=True)
        raise SystemExit(1)

    selected = results[result_number - 1]
    click.echo(f'\nSelected: {selected["title"]} ({selected["duration_str"]})')

    # Download
    click.echo('Downloading...')
    try:
        sample_path, size_mb = download_youtube(selected['url'])
        click.echo(f'  Done ({size_mb:.1f} MB)')
    except RuntimeError as e:
        click.echo(f'Download failed: {e}', err=True)
        raise SystemExit(1)

    # Derive beat name from title if not specified
    if name is None:
        clean = re.sub(r'[^\w\s]', '', selected['title'])
        words = clean.split()[:3]
        name = '_'.join(words) + f'_{genre.title()}'

    # Generate
    click.echo(f'Generating {genre} beat: "{name}"...\n')
    try:
        result = render_beat(
            sample_path, name, genre=genre, bpm=bpm, bars=bars,
            loop_start=loop_start, loop_end=loop_end,
            drums_json=drums, vinyl_slow=vinyl_slow,
            no_bass=no_bass, lofi=lofi,
            output_dir=output, kit_dir=kit,
        )
        click.echo(f'\nDone!')
    except Exception as e:
        click.echo(f'\nRender failed: {e}', err=True)
        raise SystemExit(1)


@main.command()
def genres():
    """List available genres."""
    click.echo('\nAvailable genres:\n')
    for name, desc in GENRES.items():
        click.echo(f'  {name:20s} {desc}')


@main.command()
def presets():
    """List available drum preset JSONs."""
    preset_dir = os.path.join(os.path.dirname(__file__), 'presets')
    click.echo('\nDrum presets:\n')
    if not os.path.exists(preset_dir):
        click.echo('  No presets directory found.')
        return
    for f in sorted(os.listdir(preset_dir)):
        if f.endswith('.json'):
            click.echo(f'  {f}')
    click.echo(f'\nUse: sampleflip generate <n> --genre <genre> --drums <preset.json>')


if __name__ == '__main__':
    main()
