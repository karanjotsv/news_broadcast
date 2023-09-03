import os
import sys
import subprocess


base = '/content/drive/MyDrive/GSoC2023'

def get_duration(video_f):
    """
    returns duration of a video
    """
    res = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                          "default=noprint_wrappers=1:nokey=1", video_f],
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    return float(res.stdout)


def get_frames_ffmpeg(video_f, out_path=base + '/frames', frames=384):
    """
    extracts frames from video using ffmpeg with subprocess module
    """
    f_name, ext = os.path.splitext(video_f)
    duration = get_duration(video_f)

    fps = frames/duration
    # fps = 25
    frames = int(fps*duration // 1)

    subprocess.call(['ffmpeg', '-i', video_f, '-r', str(fps), '-vframes', str(frames), f"{out_path}/%03d.jpg"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)

    return out_path
