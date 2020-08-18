from __future__ import unicode_literals
import youtube_dl


def download_yt(url, fn, sample_rate):
    ydl_opts = {
        'outtmpl': fn,
        'format': 'bestaudio/best',
        'keepvideo': False
    }

    video_title = None
    video_id = None
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        ydl.download([url])
        video_title = info_dict.get('title', None)
        video_id = info_dict.get('id', None)
    return video_title, video_id
