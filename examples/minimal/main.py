"""Minimal example submission: copy the provided submission_format.csv to submission.csv."""

import shutil
from pathlib import Path

DATA_DIR = Path("/code_execution/data")
SUBMISSION_FORMAT_CSV = DATA_DIR / "submission_format.csv"
SUBMISSION_PATH = Path("/code_execution/submission/submission.csv")


def main() -> None:
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SUBMISSION_FORMAT_CSV, SUBMISSION_PATH)
    print(f"Copied {SUBMISSION_FORMAT_CSV} to {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
