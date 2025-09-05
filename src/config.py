#from pydantic import BaseSettings
from pydantic_settings import BaseSettings
from functools import lru_cache
import torch

class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8080
    DEBUG: bool = False
    
    # Separation
    SEPARATION_ENABLED: bool = True
    SEPARATION_METHOD: str = "demucs"  # demucs | none
    DEMUCS_MODEL: str = "htdemucs"     # good default for vocals
    DEMUCS_TWO_STEMS: str = "vocals"   # we want vocals stem

    # ASR
    WHISPER_MODEL: str = "small"  # tiny|base|small|medium|large-v3; small ~= good CPU tradeoff
    ASR_DEVICE: str = "auto"      # auto|cpu|cuda
    ASR_COMPUTE_TYPE: str = "int8"  # int8|int8_float16|float16|float32
    ASR_BEAM_SIZE: int = 5
    ASR_LANGUAGE_HINT: str | None = None

    # Runtime
    MAX_FILE_MB: int = 500
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    s = Settings()
    if s.ASR_DEVICE == "auto":
        s.ASR_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return s

