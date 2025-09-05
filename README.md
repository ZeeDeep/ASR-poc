# Speech Separation + ASR POC

Containerized microservice that:
1) Accepts an audio file
2) Separates **vocals/speech** from background (Demucs; fallback to `ffmpeg afftdn`)
3) Transcribes speech using **faster-whisper**
4) Returns rich JSON with segments, timings, and pipeline info

## Why these components?

- **Demucs (htdemucs)**: state-of-the-art music source separation; its `--two-stems=vocals` reliably isolates voice.
- **Fallback (ffmpeg `afftdn`)**: fast, CPU-only spectral denoise when separation is off or fails.
- **faster-whisper**: CTranslate2 inference; **fast on CPU**, auto-uses **CUDA** if available, good accuracy/speed trade-offs (e.g., `small`).

## Quickstart

### 0) Prereqs
- Docker + docker-compose

### 1) Build & run
```bash
#docker compose build
#docker compose up

docker build --no-cache -t speech-sep-asr-poc .

docker run -it --rm -p 8080:8080 speech-sep-asr-poc

