"""Transcribe a few clips with a model from the shared Hugging Face cache.

The pod mounts the cache read-only at /code_execution/huggingface_models, with one
directory per model repo ID, so the submission zip ships no model files of its own.
"""

import os
from pathlib import Path

import librosa
import polars as pl
import torch

# The pod has no internet. Offline mode stops the hub check that would otherwise
# wait for a timeout.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from transformers import pipeline

MODEL_DIR = Path("/code_execution/huggingface_models/openai/whisper-tiny")
DATA_DIR = Path("/code_execution/data")
SUBMISSION_FORMAT_CSV = DATA_DIR / "submission_format.csv"
CLIPS_DIR = DATA_DIR / "clips"
SUBMISSION_PATH = Path("/code_execution/submission/submission.csv")
SAMPLING_RATE = 16000
N_CLIPS = 3


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # The cache holds the whole model repo, TensorFlow weights included, so name the
    # framework instead of letting transformers guess from the files present.
    asr = pipeline(
        "automatic-speech-recognition",
        model=str(MODEL_DIR),
        framework="pt",
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device=device,
    )

    submission_format = pl.read_csv(SUBMISSION_FORMAT_CSV)
    transcripts = {}
    for filename in submission_format["audio_filename"].head(N_CLIPS):
        audio, _ = librosa.load(str(CLIPS_DIR / filename), sr=SAMPLING_RATE, mono=True)
        transcripts[filename] = asr(audio)["text"].strip()

    # Clips we did not transcribe keep the placeholder text from the submission format.
    submission = submission_format.with_columns(
        pl.col("audio_filename")
        .replace_strict(transcripts, default=None)
        .fill_null(pl.col("transcript"))
        .alias("transcript")
    )
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.write_csv(SUBMISSION_PATH)
    print(f"Wrote {submission.height} predictions to {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
