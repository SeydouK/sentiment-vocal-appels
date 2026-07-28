import time
import logging

from app.audio_utils import load_and_preprocess_audio, AudioProcessingError
from app.asr import get_asr_model
from app.sentiment import get_sentiment_model
from app.config import TARGET_SAMPLE_RATE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pipeline")


class PipelineError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code 


def warmup_models() -> None:

    logger.info("Chargement des modeles ASR et sentiment.")
    get_asr_model()
    get_sentiment_model()
    logger.info("Warmup termine. Modeles prets.")


def run_pipeline(file_path: str) -> dict:
    start_time = time.time()

    try:
        signal = load_and_preprocess_audio(file_path)
    except AudioProcessingError as e:
        logger.warning(f"Erreur de pretraitement audio : {e}")
        raise PipelineError(str(e), status_code=400)
    except Exception as e:
        logger.error(f"Erreur inattendue au pretraitement : {e}")
        raise PipelineError("Erreur interne lors du traitement audio.", status_code=500)

    try:
        asr_model = get_asr_model()
        transcription = asr_model.transcribe(signal, sample_rate=TARGET_SAMPLE_RATE)
    except Exception as e:
        logger.error(f"Erreur ASR : {e}")
        raise PipelineError("Erreur interne lors de la transcription audio.", status_code=500)

    if not transcription.strip():
        logger.warning("Transcription vide. L'audio probablement incomprehensible.")

    try:
        sentiment_model = get_sentiment_model()
        sentiment_result = sentiment_model.predict(transcription)
    except Exception as e:
        logger.error(f"Erreur analyse sentiment : {e}")
        raise PipelineError("Erreur interne lors de l'analyse de sentiment.", status_code=500)

    elapsed = round(time.time() - start_time, 2)
    logger.info(f"Pipeline termine en {elapsed} -> sentiment={sentiment_result['sentiment']}")

    return {
        "transcription": transcription,
        "sentiment": sentiment_result["sentiment"],
        "confidence": sentiment_result["confidence"],
        "raw_label": sentiment_result["raw_label"],
        "processing_time_seconds": elapsed,
    }