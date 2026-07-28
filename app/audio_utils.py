import os 
import numpy as np 
import librosa 

from app.config import (
    TARGET_SAMPLE_RATE,
    MAX_AUDIO_DURATION_SECONDS,
    ALLOWED_AUDIO_EXTENSIONS,
)

class AudioProcessingError(Exception): 
    pass 

def validate_extension(filename: str) -> None: 
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise AudioProcessingError(
            f"Format non supporté : '{ext}'. "
            f"Formats acceptés : {', '.join(ALLOWED_AUDIO_EXTENSIONS)}"
        )

def load_and_preprocess_audio(file_path:str) -> np.ndarray:
    validate_extension(file_path)

    if not os.path.exists(file_path):
        raise AudioProcessingError(f"Fichier introuvable: {file_path}")

    if os.path.getsize(file_path) == 0:
        raise AudioProcessingError("Le fichier audio est vide.")

    try: 
        signal, original_sr = librosa.load(file_path, sr=None, mono=False)
    except Exception as e: 
        raise AudioProcessingError(f"Impossible de lire le fichier audioL {e}.")
    
    if signal is None or signal.size == 0:
        raise AudioProcessingError("Le fichier ne contient aucune donnees audio exploitable.")

    if signal.ndim > 1:
        signal = librosa.to_mono(signal)

    duration_sec = len(signal) / original_sr
    if duration_sec > MAX_AUDIO_DURATION_SECONDS:
        raise AudioProcessingError(
            f"Durée audio ({duration_sec:.1f}s) dépasse le maximum autorisé "
            f"({MAX_AUDIO_DURATION_SECONDS}s / 5 min)."
        )

    if original_sr != TARGET_SAMPLE_RATE:
        signal = librosa.resample(
            signal, orig_sr = original_sr, target_sr = TARGET_SAMPLE_RATE
        )

    rms = np.sqrt(np.mean(signal**2))
    if rms< 1e-4:
        raise AudioProcessingError(
            "Aucun signal exploitable detecte."
        )