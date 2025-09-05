import os
import time
import uuid
import json
from typing import Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .audio_utils import to_wav_mono16k
from .separation import separate_or_denoise
from .asr import transcribe
from .schemas import TranscribeResponse, PipelineOut, TimingsOut
from .diarization import Diarizer   # <-- NEW

app = FastAPI(title="Speech Separation + ASR + Diarization POC", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# global diarizer instance (lazy-loaded)
diarizer = None


@app.post("/v1/transcribe", response_model=TranscribeResponse)
async def v1_transcribe(
    file: UploadFile = File(..., description="audio file"),
    params_json: Optional[str] = Form(default=None, description="JSON params"),
):
    """
    Multipart form:
      - file: audio file (wav/mp3/m4a/flac/ogg)
      - params_json: optional JSON string:
          { "model_size": "small",
            "language_hint": "en",
            "separation_enabled": true,
            "separation_method": "demucs",
            "diarization": true }
    """
    s = get_settings()
    t0 = time.perf_counter()

    # --- validate & save input ---
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > s.MAX_FILE_MB:
        return JSONResponse(
            status_code=400,
            content={"detail": f"File too large (> {s.MAX_FILE_MB} MB)"}
        )
    tmp_in = f"/tmp/{uuid.uuid4()}_{file.filename}"
    with open(tmp_in, "wb") as f:
        f.write(contents)

    # --- parse params ---
    user_params = {}
    if params_json:
        try:
            user_params = json.loads(params_json)
        except Exception:
            return JSONResponse(
                status_code=400,
                content={"detail": "params_json is not valid JSON"}
            )

    separation_enabled = user_params.get("separation_enabled", s.SEPARATION_ENABLED)
    separation_method = user_params.get("separation_method", s.SEPARATION_METHOD)
    model_size = user_params.get("model_size", s.WHISPER_MODEL)
    language_hint = user_params.get("language_hint", s.ASR_LANGUAGE_HINT)
    diarization_flag = bool(user_params.get("diarization", False))

    # --- step 1: decode/transcode ---
    t_load0 = time.perf_counter()
    wav16k, sr, dur = to_wav_mono16k(tmp_in)
    t_load = int((time.perf_counter() - t_load0) * 1000)

    # --- step 2: separation (or fallback) ---
    t_sep0 = time.perf_counter()
    processed_wav, sep_meta = separate_or_denoise(
        wav16k, separation_enabled, separation_method
    )
    t_sep = int((time.perf_counter() - t_sep0) * 1000)

    # --- step 3: ASR ---
    t_asr0 = time.perf_counter()
    segments, text, language = transcribe(processed_wav, model_size, language_hint)
    t_asr = int((time.perf_counter() - t_asr0) * 1000)

    # --- step 4: Diarization (optional) ---
    diarization_segments = None
    t_diar = 0
    if diarization_flag:
        global diarizer
        if diarizer is None:
            diarizer = Diarizer()
        t_diar0 = time.perf_counter()
        diarization_segments = diarizer.run(processed_wav)
        t_diar = int((time.perf_counter() - t_diar0) * 1000)

        # merge speaker labels into ASR segments (basic alignment by time)
        for seg in segments:
            seg_start, seg_end = seg.get("start"), seg.get("end")
            speaker = None
            for d in diarization_segments:
                if seg_start >= d["start"] and seg_end <= d["end"]:
                    speaker = d["speaker"]
                    break
            if speaker:
                seg["speaker"] = speaker

    # --- timings ---
    total_ms = int((time.perf_counter() - t0) * 1000)
    request_id = str(uuid.uuid4())
    pipeline = PipelineOut(
        separation=sep_meta,
        transcription={"model": model_size, "device": s.ASR_DEVICE}
    )
    if diarization_flag:
        pipeline.diarization = {"enabled": True, "method": "speechbrain"}

    timings = TimingsOut(
        load=t_load, separation=t_sep,
        transcription=t_asr, total=total_ms
    )
    if diarization_flag:
        timings.diarization = t_diar

    # --- cleanup ---
    for p in [tmp_in, wav16k, processed_wav]:
        try:
            os.remove(p)
        except Exception:
            pass

    # --- response ---
    return TranscribeResponse(
        request_id=request_id,
        duration_sec=dur,
        sample_rate=sr,
        pipeline=pipeline,
        segments=segments,
        text=text,
        language=language,
        timings_ms=timings,
        diarization_segments=diarization_segments,
    )


if __name__ == "__main__":
    uvicorn.run("src.app:app", host="0.0.0.0", port=8080, reload=False)

