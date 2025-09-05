from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class TranscribeParams(BaseModel):
    separation_enabled: Optional[bool] = None
    separation_method: Optional[str] = None
    model_size: Optional[str] = None
    language_hint: Optional[str] = None
    diarization: Optional[bool] = False

class SegmentOut(BaseModel):
    start: float
    end: float
    text: str

class PipelineOut(BaseModel):
    separation: Dict
    transcription: Dict

class TimingsOut(BaseModel):
    load: int
    separation: int
    transcription: int
    total: int

class TranscribeResponse(BaseModel):
    request_id: str
    duration_sec: float
    sample_rate: int
    pipeline: PipelineOut
    segments: List[SegmentOut]
    text: str
    language: str
    timings_ms: TimingsOut

