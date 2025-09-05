from fastapi.testclient import TestClient
from src.app import app
import io

client = TestClient(app)

def test_reject_large():
    big = io.BytesIO(b"\x00" * (60 * 1024 * 1024))
    files = {"file": ("big.wav", big, "audio/wav")}
    r = client.post("/v1/transcribe", files=files)
    assert r.status_code in (200, 400)
    if r.status_code == 400:
        assert "File too large" in r.json()["detail"]

