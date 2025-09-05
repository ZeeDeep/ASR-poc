import torch
from speechbrain.pretrained import SpeakerDiarization

class Diarizer:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Load pretrained diarization model
        self.pipeline = SpeakerDiarization.from_hparams(
            source="speechbrain/diarization-vad-crdnn-libriparty",
            savedir="pretrained_models/diarization",
            run_opts={"device": self.device}
        )

    def run(self, audio_file: str):
        """
        Run diarization on an audio file.
        Returns list of dicts: [{"start": float, "end": float, "speaker": str}]
        """
        diarization = self.pipeline(audio_file)
        segments = []

        for seg in diarization.get_timeline():
            spk = diarization[seg]
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "speaker": spk
            })

        return segments

