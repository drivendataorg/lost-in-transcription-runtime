"""Submission template: loop over submission_format.csv and write one transcription per clip."""

from pathlib import Path

import polars as pl

DATA_DIR = Path("/code_execution/data")
SUBMISSION_FORMAT_CSV = DATA_DIR / "submission_format.csv"
CLIPS_DIR = DATA_DIR / "clips"
SUBMISSION_PATH = Path("/code_execution/submission/submission.csv")


def transcribe(audio_path: Path) -> str:
    """Return the transcription for one audio clip.

    Replace this function body with your model.
    """
    raise NotImplementedError


def main() -> None:
    submission = pl.read_csv(SUBMISSION_FORMAT_CSV)

    transcriptions = [transcribe(CLIPS_DIR / filename) for filename in submission["audio_filename"]]

    submission = submission.with_columns(pl.Series("transcript", transcriptions))
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.write_csv(SUBMISSION_PATH)
    print(f"Wrote {submission.height} predictions to {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
