# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "jiwer",
#     "pandas",
#     "typer",
# ]
# ///
import re
from pathlib import Path

import jiwer
import pandas as pd
import typer


def _lowercase_sentence_initial(match):
    """Lowercase a sentence-initial letter.

    Keeps the letter uppercase when the next letter is also uppercase, which marks
    the start of an acronym or initialism.

    Input: a `re.Match` from the SENTENCE_INITIAL pattern with three groups:
        1. sentence delimiter (start-of-string or `.!?—` plus optional space)
        2. first letter of the sentence
        3. next letter (may be empty)
    Output: the joined string with group 2 possibly lowercased.
    """
    delimiter, first, second = match.group(1), match.group(2), match.group(3)
    if first.isupper() and not (second and second.isupper()):
        first = first.lower()
    return delimiter + first + second


def normalize_text(text):
    """Normalize a transcript to a canonical form for scoring.

    Removes bracketed annotations (`[...]`), unintelligible markers (`(?)`),
    parenthetical wrappers, and stray punctuation (`¿¡";:!?`). Lowercases
    sentence-initial letters. Rewrites em dashes as commas, then drops commas
    and periods (but preserves `...`). Collapses runs of whitespace.

    Input: a raw transcript string.
    Output: the cleaned transcript string.
    """
    BRACKETED = re.compile(r"\[[^\]]+\]")
    UNINTELLIGIBLE_PAREN = re.compile(r"\(\?+\)")
    WORD_PAREN = re.compile(r"\(([^()]*)\)")
    PUNCTUATION_OTHER = re.compile('[¿¡";:]+')
    COMMA = re.compile(",+")
    # [^\W\d_] matches any Unicode letter (word char minus digits/underscore),
    # including sentence-initial accented capitals (e.g. Spanish Á/É/Í/Ó/Ú/Ñ)
    SENTENCE_INITIAL = re.compile(r"(^\s*|[.!?—]\s*)([^\W\d_])([^\W\d_]?)")
    SENTENCE_END = re.compile("[!?]+")
    MULTISPACE = re.compile("  +")

    text = text.replace("~", "")
    text = re.sub(BRACKETED, " ", text)
    text = re.sub(UNINTELLIGIBLE_PAREN, " ", text)
    text = re.sub(WORD_PAREN, r"\1", text)
    text = text.replace("#x27;", "'")
    text = re.sub(PUNCTUATION_OTHER, " ", text)
    text = re.sub(SENTENCE_INITIAL, _lowercase_sentence_initial, text)
    # self-interruption em dash becomes a comma + space, same as any other comma, so it
    # collapses away to a single space by the time normalization finishes
    text = text.replace("—", ", ")
    text = re.sub(COMMA, " ", text)
    text = re.sub(SENTENCE_END, " ", text)
    text = text.replace("...", "!ELLIPSIS!").replace(".", " ").replace("!ELLIPSIS!", "...")
    while " ... " in text:
        text = text.replace(" ... ", " ")
    text = re.sub(MULTISPACE, " ", text)
    return text


def _word_error_rate(predicted, actual, normalize_function):
    """Compute corpus-level word error rate between predicted and reference transcripts.

    Runs `normalize_function` on every transcript before comparison.

    Input:
        predicted: 2-D array of shape (n_samples, 1) holding predicted transcripts.
        actual: 2-D array of shape (n_samples, 1) holding reference transcripts.
        normalize_function: callable that takes a transcript string and returns a
            cleaned string.
    Output: WER as a float in [0, ∞), computed across all samples.
    """
    normalized_preds = [normalize_function(text) for text in predicted[:, 0]]
    normalized_refs = [normalize_function(text) for text in actual[:, 0]]
    return jiwer.wer(normalized_refs, normalized_preds)


def main(
    actual_path: Path = typer.Argument(..., help="Path to the ground truth CSV."),
    predicted_path: Path = typer.Option(
        Path("runtime/submission/submission.csv"),
        help="Path to the submission CSV.",
    ),
):
    """Score a submission CSV against a ground truth CSV.

    Both files must have columns `audio_filename` and `transcript`. Predicted
    rows are aligned to the ground truth by `audio_filename`; a missing
    prediction is an error. Prints the corpus-level word error rate to stdout.
    """
    predicted = pd.read_csv(predicted_path).set_index("audio_filename").sort_index()
    actual = pd.read_csv(actual_path).set_index("audio_filename").sort_index()

    missing = actual.index.difference(predicted.index)
    if len(missing):
        raise typer.BadParameter(
            f"Submission is missing {len(missing)} rows from ground truth "
            f"(first: {missing[0]})."
        )
    predicted = predicted.loc[actual.index]

    wer = _word_error_rate(
        predicted[["transcript"]].fillna("").to_numpy(),
        actual[["transcript"]].fillna("").to_numpy(),
        normalize_text,
    )
    typer.echo(f"WER: {wer:.6f}")


if __name__ == "__main__":
    typer.run(main)
