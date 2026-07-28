import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from app.config import ASR_MODEL_NAME


class ASRModel:
    def __init__(self, model_name: str = ASR_MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[ASR] Chargement du modèle '{model_name}' sur {self.device}...")

        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        self.model.eval()  

        print("[ASR] Modèle chargé avec succès.")

    def transcribe(self, audio_signal: np.ndarray, sample_rate: int = 16000) -> str:
        inputs = self.processor(
            audio_signal,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )

        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            logits = self.model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)

        transcription = self.processor.batch_decode(predicted_ids)[0]

        return transcription.strip().lower()

_asr_model_instance = None

def get_asr_model() -> ASRModel:
    global _asr_model_instance
    if _asr_model_instance is None:
        _asr_model_instance = ASRModel()
    return _asr_model_instance