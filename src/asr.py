from faster_whisper import WhisperModel
from functools import lru_cache
from .config import get_settings

@lru_cache()
def _load_model(model_size: str, device: str, compute_type: str) -> WhisperModel:
    # Loads once per process
    return WhisperModel(model_size, device=device, compute_type=compute_type)

def transcribe(audio_path: str, model_size: str | None, language_hint: str | None):
    s = get_settings()
    size = model_size or s.WHISPER_MODEL
    model = _load_model(size, s.ASR_DEVICE, s.ASR_COMPUTE_TYPE)

    segments, info = model.transcribe(
        audio_path,
        language=language_hint or s.ASR_LANGUAGE_HINT,
        vad_filter=True,            # improves robustness on silence/noise
        vad_parameters=dict(min_silence_duration_ms=300),
        beam_size=s.ASR_BEAM_SIZE,
        word_timestamps=False,
    )

    out_segments = []
    full_text_parts = []
    for seg in segments:
        out_segments.append(dict(start=float(seg.start), end=float(seg.end), text=seg.text.strip()))
        full_text_parts.append(seg.text.strip())

    full_text = " ".join(full_text_parts).strip()
    return out_segments, full_text, (info.language or "unknown")

