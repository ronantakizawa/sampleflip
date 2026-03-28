"""YouTube search via yt-dlp — list results without downloading."""

import json
import os
import subprocess
import tempfile

CACHE_PATH = os.path.join(tempfile.gettempdir(), 'sampleflip_results.json')


def search_youtube(query, count=5):
    """Search YouTube and return list of results (no download).

    Returns list of dicts: {id, title, duration, channel, url}
    """
    cmd = [
        'yt-dlp',
        f'ytsearch{count}:{query}',
        '--flat-playlist',
        '--dump-json',
        '--no-download',
        '--quiet', '--no-warnings',
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        raise RuntimeError('yt-dlp not found. Install: brew install yt-dlp')
    except subprocess.TimeoutExpired:
        raise RuntimeError('YouTube search timed out')

    if result.returncode != 0:
        raise RuntimeError(f'yt-dlp error: {result.stderr[:200]}')

    results = []
    for line in result.stdout.strip().split('\n'):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            dur = data.get('duration') or 0
            mins, secs = int(dur // 60), int(dur % 60)
            results.append({
                'id': data.get('id', ''),
                'title': data.get('title', 'Unknown'),
                'duration': dur,
                'duration_str': f'{mins}:{secs:02d}',
                'channel': data.get('channel', data.get('uploader', 'Unknown')),
                'url': data.get('url') or f'https://www.youtube.com/watch?v={data.get("id", "")}',
            })
        except json.JSONDecodeError:
            continue

    # Cache results for generate command
    with open(CACHE_PATH, 'w') as f:
        json.dump({'query': query, 'results': results}, f)

    return results


def get_cached_results():
    """Load cached search results."""
    if not os.path.exists(CACHE_PATH):
        return None
    with open(CACHE_PATH) as f:
        return json.load(f)
