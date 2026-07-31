# Détection de Sentiment dans les Appels Vocaux

Pipeline qui transcrit un appel vocal (audio) puis détecte le sentiment du client : **positif**, **négatif** ou **neutre**.

Projet réalisé pour le module Deep Learning 2 (Master 2 IA) Dakar Institute of Technology.

## Architecture

```
Audio (.wav ou .mp3) → Prétraitement (mono, 16kHz) → ASR (Wav2Vec2.0) → Sentiment (CamemBERT) → Résultat JSON
```

- **ASR** : `jonatasgrosman/wav2vec2-large-xlsr-53-french` — transcription voix → texte en français.
- **Sentiment** : `cmarkea/distilcamembert-base-sentiment` — CamemBERT distillé, sort 5 classes (1 à 5 étoiles), mappées vers 3 classes :
  - 1-2 étoiles → négatif
  - 3 étoiles → neutre
  - 4-5 étoiles → positif

Le même pipeline (`app/pipeline.py`) est utilisé par l'API REST et par l'interface Gradio, pour éviter toute duplication de logique.

## Structure du projet

```
app/
├── audio_utils.py   # prétraitement audio + gestion des erreurs
├── asr.py            # transcription avec Wav2Vec2.0
├── sentiment.py       # analyse de sentiment avec CamemBERT
├── pipeline.py         # assemblage complet
└── api.py               # API FastAPI 
gradio_app.py              # interface graphiquu
tests/
└── audio_samples/         # 3 fichiers de démo (positif/négatif/neutre)
```

## Installation

```bash
git clone https://github.com/SeydouK/sentiment-vocal-appels.git
cd sentiment-vocal-appels

python -m venv .venv
.\.venv\Scripts\activate.bat

pip install -r requirements.txt
```

## Utilisation

### Interface Gradio

```bash
python gradio_app.py
```
Ouvrez ensuite `http://127.0.0.1:7860` dans le navigateur. L'application vous donne la possibilite de Upload un fichier audio ou clique sur un exemple.

### API REST

```bash
uvicorn main:app --reload --port 8000
```
Documentation interactive : `http://localhost:8000/docs`

Exemple d'appel avec curl :
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@tests/audio_samples/positif.wav"
```

Réponse :
```json
{
  "transcription": "je suis très content du service",
  "sentiment": "positif",
  "confidence": 0.87,
  "raw_label": "5 stars",
  "processing_time_seconds": 2.31
}
```

### Tests

```bash
python -m pytest tests/ -v
```

## Cas d'usage

Analyser automatiquement des enregistrements d'appels du service client pour détecter les clients mécontents sans écoute manuelle — priorisation du support, détection de tendances, contrôle qualité.

## Limites connues

- ASR non ponctué et en minuscules peut affecter la précision du sentiment sur des phrases ambiguës.
- Testé uniquement sur du français ; performance non garantie sur d'autres langues ou accents forts.
- Fichiers audio limités à 5 minutes et aux formats `.wav`/`.mp3`.
