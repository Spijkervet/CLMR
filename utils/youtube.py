from __future__ import unicode_literals
import youtube_dl


def download_yt(url, fn, sample_rate):
    ydl_opts = {
        'outtmpl': fn,
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
            'postprocessor_args': [
            '-ar', str(sample_rate)
        ],
        'prefer_ffmpeg': True,
        'keepvideo': False
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])