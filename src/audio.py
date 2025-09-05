import subprocess, tempfile, os, json, shlex
from pathlib import Path

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

def run(cmd: str) -> None:
    proc = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))

def ffprobe_sr_dur(path: str) -> tuple[int, float]:
    cmd = f'{FFPROBE} -v error -select_streams a:0 -show_entries stream=sample_rate,duration -of json "{path}"'
    proc = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode())
    data = json.loads(proc.stdout.decode())
    sr = int(data["streams"][0]["sample_rate"])
    dur = float(data["streams"][0]["duration"])
    return sr, dur

def to_wav_mono16k(src_path: str) -> tuple[str, int, float]:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    out = tmp.name
    cmd = f'{FFMPEG} -y -i "{src_path}" -ac 1 -ar 16000 -c:a pcm_s16le "{out}"'
    run(cmd)
    sr, dur = ffprobe_sr_dur(out)
    return out, sr, dur

def denoise_ffmpeg_afftdn(src_wav: str) -> str:
    """Lightweight noise suppression fallback (CPU-friendly)."""
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    cmd = f'{FFMPEG} -y -i "{src_wav}" -af "afftdn=nr=12" -c:a pcm_s16le "{out}"'
    run(cmd)
    return out

