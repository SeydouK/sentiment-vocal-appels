import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.pipeline import run_pipeline, warmup_models
from sklearn.metrics import accuracy_score, f1_score, classification_report

TEST_SET = [
    {"file": "tests/audio_samples/positif.wav", "true_sentiment": "positif",
     "reference_text": None},  # mets ta transcription exacte ici si tu veux le WER
    {"file": "tests/audio_samples/negatif.wav", "true_sentiment": "negatif",
     "reference_text": None},
    {"file": "tests/audio_samples/neutre.wav", "true_sentiment": "neutre",
     "reference_text": None},
]


def compute_wer(reference: str, hypothesis: str) -> float:
    """Calcule le Word Error Rate entre une transcription de référence et la prédiction.
    Implémentation simple par distance de Levenshtein au niveau mot.
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])

    return d[len(ref_words)][len(hyp_words)] / max(len(ref_words), 1)


def main():
    print("Chargement des modèles en cous...\n")
    warmup_models()

    y_true, y_pred = [], []
    wer_scores = []

    for item in TEST_SET:
        result = run_pipeline(item["file"])
        predicted = result["sentiment"]

        y_true.append(item["true_sentiment"])
        y_pred.append(predicted)

        print(f"Fichier : {item['file']}")
        print(f"  Transcription     : {result['transcription']}")
        print(f"  Sentiment attendu : {item['true_sentiment']}")
        print(f"  Sentiment prédit  : {predicted} ({result['confidence']*100:.1f}%)")

        if item["reference_text"]:
            wer = compute_wer(item["reference_text"], result["transcription"])
            wer_scores.append(wer)
            print(f"  WER               : {wer:.2%}")

        print()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    print("=" * 50)
    print("RÉSUMÉ ÉVALUATION")
    print("=" * 50)
    print(f"Accuracy (sentiment) : {acc:.2%}")
    print(f"F1-score macro        : {f1:.2%}")
    print()
    print(classification_report(y_true, y_pred, zero_division=0))

    if wer_scores:
        print(f"WER moyen (ASR) : {sum(wer_scores)/len(wer_scores):.2%}")
    else:
        print("WER non calculé : ajoute 'reference_text' dans TEST_SET pour l'obtenir.")


if __name__ == "__main__":
    main()