"""Download YouTube audio — tries Cobalt API first, falls back to yt-dlp."""

import os
import subprocess
import tempfile
import requests


def _download_cobalt(url, output_dir):
    """Download audio via Cobalt API (works from cloud servers)."""
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
        raise RuntimeError(f'Cobalt API error: {resp.status_code} {resp.text[:200]}')

    data = resp.json()
    download_url = data.get('url')
    if not download_url:
        raise RuntimeError(f'Cobalt returned no URL: {data}')

    # Download the audio file
    audio_resp = requests.get(download_url, stream=True, timeout=120)
    if audio_resp.status_code != 200:
        raise RuntimeError(f'Audio download failed: {audio_resp.status_code}')

    # Save to file
    fname = 'cobalt_audio.mp3'
    fpath = os.path.join(output_dir, fname)
    with open(fpath, 'wb') as f:
        for chunk in audio_resp.iter_content(chunk_size=8192):
            f.write(chunk)

    size_mb = os.path.getsize(fpath) / 1e6
    return fpath, size_mb


def _download_ytdlp(url, output_dir):
    """Download audio via yt-dlp (works locally, blocked on some cloud servers)."""
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
    """Download audio from YouTube URL. Tries Cobalt API first, falls back to yt-dlp."""
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='sampleflip_')
    os.makedirs(output_dir, exist_ok=True)

    # Try Cobalt first (works on cloud servers)
    try:
        return _download_cobalt(url, output_dir)
    except Exception as e:
        print(f'  Cobalt failed ({e}), trying yt-dlp...')

    # Fall back to yt-dlp (works locally)
    return _download_ytdlp(url, output_dir)
