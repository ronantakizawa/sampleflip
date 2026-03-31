"""Download YouTube audio — tries multiple methods."""

import os
import subprocess
import tempfile
import requests


INVIDIOUS_INSTANCES = [
    'inv.nadeko.net',
    'invidious.perennialte.ch',
    'invidious.privacyredirect.com',
    'iv.datura.network',
    'invidious.protokolla.fi',
]


def _extract_video_id(url):
    """Extract YouTube video ID from various URL formats."""
    import re
    patterns = [
        r'(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'(?:shorts/)([a-zA-Z0-9_-]{11})',
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def _download_invidious(url, output_dir):
    """Download audio via Invidious API instances."""
    video_id = _extract_video_id(url)
    if not video_id:
        raise RuntimeError(f'Could not extract video ID from: {url}')

    for instance in INVIDIOUS_INSTANCES:
        try:
            r = requests.get(
                f'https://{instance}/api/v1/videos/{video_id}',
                timeout=15,
            )
            if r.status_code != 200:
                continue

            data = r.json()
            audio_streams = [
                f for f in data.get('adaptiveFormats', [])
                if f.get('type', '').startswith('audio/')
            ]
            if not audio_streams:
                continue

            # Pick highest bitrate
            audio_streams.sort(key=lambda x: int(x.get('bitrate', '0')), reverse=True)
            stream = audio_streams[0]
            audio_url = stream['url']

            # Download
            audio_resp = requests.get(audio_url, stream=True, timeout=120)
            if audio_resp.status_code != 200:
                continue

            ext = 'webm' if 'webm' in stream.get('type', '') else 'm4a'
            fpath = os.path.join(output_dir, f'invidious_audio.{ext}')
            with open(fpath, 'wb') as f:
                for chunk in audio_resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            size_mb = os.path.getsize(fpath) / 1e6
            print(f'  Downloaded via {instance} ({size_mb:.1f} MB)')
            return fpath, size_mb

        except Exception as e:
            print(f'  Invidious {instance} failed: {e}')
            continue

    raise RuntimeError('All Invidious instances failed')


def _download_cobalt(url, output_dir):
    """Download audio via Cobalt API."""
    resp = requests.post(
        'https://api.cobalt.tools/',
        headers={
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        },
        json={
            'url': url,
            'downloadMode': 'audio',
            'audioFormat': 'mp3',
            'audioBitrate': '320',
        },
        timeout=30,
    )

    if resp.status_code != 200:
        raise RuntimeError(f'Cobalt error: {resp.status_code} {resp.text[:200]}')

    data = resp.json()
    download_url = data.get('url')
    if not download_url:
        raise RuntimeError(f'No URL in response: {data}')

    audio_resp = requests.get(download_url, stream=True, timeout=120)
    if audio_resp.status_code != 200:
        raise RuntimeError(f'Download failed: {audio_resp.status_code}')

    fpath = os.path.join(output_dir, 'cobalt_audio.mp3')
    with open(fpath, 'wb') as f:
        for chunk in audio_resp.iter_content(chunk_size=8192):
            f.write(chunk)

    size_mb = os.path.getsize(fpath) / 1e6
    return fpath, size_mb


def _download_ytdlp(url, output_dir):
    """Download audio via yt-dlp (works locally)."""
    output_template = os.path.join(output_dir, '%(title).60s.%(ext)s')

    cmd = [
        'yt-dlp', url,
        '-x', '--audio-format', 'wav',
        '--audio-quality', '0',
        '-o', output_template,
        '--no-playlist',
        '--max-filesize', '100M',
        '--socket-timeout', '30',
        '--quiet', '--no-warnings',
        '--print', 'after_move:filepath',
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        raise RuntimeError('yt-dlp not found')
    except subprocess.TimeoutExpired:
        raise RuntimeError('Download timed out')

    if result.returncode != 0:
        raise RuntimeError(f'yt-dlp failed: {result.stderr[:200]}')

    fpath = result.stdout.strip().split('\n')[-1]
    if not os.path.exists(fpath):
        raise RuntimeError(f'File not found: {fpath}')

    size_mb = os.path.getsize(fpath) / 1e6
    return fpath, size_mb


def download_youtube(url, output_dir=None):
    """Download audio from YouTube URL. Tries Invidious → Cobalt → yt-dlp."""
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='sampleflip_')
    os.makedirs(output_dir, exist_ok=True)

    errors = []

    # Try Invidious first (multiple instances)
    try:
        return _download_invidious(url, output_dir)
    except Exception as e:
        errors.append(f'Invidious: {e}')
        print(f'  Invidious failed, trying Cobalt...')

    # Try Cobalt
    try:
        return _download_cobalt(url, output_dir)
    except Exception as e:
        errors.append(f'Cobalt: {e}')
        print(f'  Cobalt failed, trying yt-dlp...')

    # Fall back to yt-dlp
    try:
        return _download_ytdlp(url, output_dir)
    except Exception as e:
        errors.append(f'yt-dlp: {e}')

    raise RuntimeError(f'All download methods failed: {"; ".join(errors)}')
