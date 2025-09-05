import os, tempfile, shutil, glob, subprocess, shlex
from .config import get_settings
from .audio import denoise_ffmpeg_afftdn

def demucs_separate(wav_path: str) -> str:
    """
    Runs demucs CLI to extract vocals stem. Returns path to vocals WAV.
    """
    s = get_settings()
    outdir = tempfile.mkdtemp(prefix="demucs_")
    cmd = (
        f'demucs -n {s.DEMUCS_MODEL} --two-stems={s.DEMUCS_TWO_STEMS} '
        f'--mp3=0 -o "{outdir}" "{wav_path}"'
    )
    proc = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        shutil.rmtree(outdir, ignore_errors=True)
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))

    # Demucs writes: {outdir}/{model}/{basename}/vocals.wav
    pattern = os.path.join(outdir, s.DEMUCS_MODEL, "*", "vocals.wav")
    matches = glob.glob(pattern)
    if not matches:
        shutil.rmtree(outdir, ignore_errors=True)
        raise RuntimeError("Demucs output not found.")
    return matches[0]

def separate_or_denoise(wav_path: str, enabled: bool, method: str) -> tuple[str, dict]:
    """
    Returns (processed_wav_path, separation_meta)
    """
    if not enabled or method == "none":
        return wav_path, {"enabled": False, "method": "none", "fallback": False}

    try:
        vocals_wav = demucs_separate(wav_path)
        return vocals_wav, {"enabled": True, "method": "demucs", "fallback": False}
    except Exception as e:
        # Fallback to denoise
        denoised = denoise_ffmpeg_afftdn(wav_path)
        return denoised, {
            "enabled": True,
            "method": "demucs",
            "fallback": True,
            "message": f"separation failed, used afftdn: {str(e)[:160]}",
        }

