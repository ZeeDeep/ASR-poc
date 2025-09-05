# Speech Separation + ASR POC

Containerized microservice that:
1) Accepts an audio file
2) Separates **vocals/speech** from background (Demucs; fallback to `ffmpeg afftdn`)
3) Transcribes speech using **faster-whisper**
4) Returns rich JSON with segments, speaker annotation, timings, and pipeline info

### 0) Prereqs
- Docker + docker-compose

### 1) Build & run
docker build --no-cache -t speech-sep-asr-poc .
docker run -it --rm -p 8080:8080 speech-sep-asr-poc

### Access API Endpoint
http://localhost:8080/docs
