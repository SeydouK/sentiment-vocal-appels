import os
import gradio as gr

from app.pipeline import run_pipeline, warmup_models, PipelineError

SENTIMENT_DISPLAY = {
    "positif": ":) Positif",
    "negatif": ">:( Négatif",
    "neutre": ":| Neutre",
}

EXAMPLES_DIR = "tests/audio_samples"


def analyze_audio(audio_file):
    if audio_file is None:
        return "", "", 0.0, "Attention! - Veuillez fournir un fichier audio."

    try:
        result = run_pipeline(audio_file)
        sentiment_label = SENTIMENT_DISPLAY.get(result["sentiment"], result["sentiment"])
        confidence_pct = round(result["confidence"] * 100, 1)

        return (
            result["transcription"],
            sentiment_label,
            confidence_pct,
            "",
        )

    except PipelineError as e:
        return "", "", 0.0, f"Attention! -  {e.message}"

    except Exception as e:
        return "", "", 0.0, f"Attention! -  Erreur inattendue : {e}"


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Detection de Sentiment Vocal") as demo:
        gr.Markdown(
            """
            #  Detection Automatique de Sentiment dans les Appels Vocaux
            Pipeline : **Wav2Vec 2.0** (transcription) → **CamemBERT** (analyse de sentiment)

            Chargez un fichier audio (.wav ou .mp3, max 5 min), sur la plateforme, ou choisissez un exemple ci-dessous.
            """
        )

        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    label="Fichier audio", type="filepath", sources=["upload", "microphone"]
                )
                submit_btn = gr.Button("Analyser", variant="primary")

                if os.path.isdir(EXAMPLES_DIR):
                    example_files = [
                        os.path.join(EXAMPLES_DIR, f)
                        for f in os.listdir(EXAMPLES_DIR)
                        if f.endswith((".wav", ".mp3"))
                    ]
                    if example_files:
                        gr.Examples(examples=example_files, inputs=audio_input, label="Exemples")

            with gr.Column():
                transcription_output = gr.Textbox(
                    label="Transcription ", lines=3, interactive=False
                )
                sentiment_output = gr.Textbox(label="Sentiment detecte", interactive=False)
                confidence_output = gr.Number(label="Score de confiance (%)", interactive=False)
                error_output = gr.Textbox(label="Erreur", interactive=False, visible=True)

        submit_btn.click(
            fn=analyze_audio,
            inputs=audio_input,
            outputs=[transcription_output, sentiment_output, confidence_output, error_output],
        )

    return demo


if __name__ == "__main__":
    print("Chargement des modeles avant lancement de l'interface en cours, veuillez patienter.")
    warmup_models()

    demo = build_interface()
    demo.launch()