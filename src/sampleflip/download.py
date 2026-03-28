"""Download a YouTube video as WAV audio."""

import os
import subprocess
import tempfile


def download_youtube(url, output_dir=None):
    """Download audio from YouTube URL as WAV.

    Returns path to downloaded WAV file.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='sampleflip_')
    os.makedirs(output_dir, exist_ok=True)

    output_template = os.path.join(output_dir, '%(title).60s.%(ext)s')

    cmd = [
        'yt-dlp',
        url,
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
        raise RuntimeError('yt-dlp not found. Install: brew install yt-dlp')
    except subprocess.TimeoutExpired:
        raise RuntimeError('Download timed out (120s limit)')

    if result.returncode != 0:
        raise RuntimeError(f'Download failed: {result.stderr[:200]}')

    fpath = result.stdout.strip().split('\n')[-1]
    if not os.path.exists(fpath):
        raise RuntimeError(f'Downloaded file not found: {fpath}')

    size_mb = os.path.getsize(fpath) / 1e6
    return fpath, size_mb
