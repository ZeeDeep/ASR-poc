from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class SegmentOut(BaseModel):
    start: float
    end: float
    text: str
    speaker: Optional[str] = None  # added for diarization


class SeparationOut(BaseModel):
    enabled: bool
    method: str
    fallback: Optional[bool] = False
    message: Optional[str] = None


class PipelineOut(BaseModel):
    separation: SeparationOut
    transcription: Dict[str, Any]
    diarization: Optional[Dict[str, Any]] = None


class TimingsOut(BaseModel):
    load: int
    separation: int
    transcription: int
    total: int
    diarization: Optional[int] = None


class DiarizationSegment(BaseModel):
    start: float
    end: float
    speaker: str


class TranscribeResponse(BaseModel):
    request_id: str
    duration_sec: float
    sample_rate: int
    pipeline: PipelineOut
    segments: List[SegmentOut]
    text: str
    language: str
    timings_ms: TimingsOut
    diarization_segments: Optional[List[DiarizationSegment]] = None


