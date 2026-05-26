"""Pick out neurons whose MILAN description mentions text/word/letter."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List

import pandas as pd

# Same set the upstream `experiments/edit.py` uses for IMAGENET_SPURIOUS_TEXT.
TARGET_WORDS = ("word", "text", "letter")


def text_neuron_mask(descriptions: Iterable[str],
                     target_words: Iterable[str] = TARGET_WORDS) -> List[bool]:
    """Return a boolean list: True iff the description contains any target word.

    Matches whole-word, case-insensitive, so 'letterhead' counts as a hit but
    'litter' does not.
    """
    pat = re.compile(r"\b(" + "|".join(re.escape(w) for w in target_words) + r")",
                     flags=re.IGNORECASE)
    return [bool(pat.search(d or "")) for d in descriptions]


def annotate(descriptions_csv: Path, out_csv: Path,
             target_words: Iterable[str] = TARGET_WORDS) -> Path:
    """Add an `is_text_neuron` column to the descriptions CSV."""
    df = pd.read_csv(descriptions_csv)
    df["is_text_neuron"] = text_neuron_mask(df["description"], target_words)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    n_text = int(df["is_text_neuron"].sum())
    print(f"{n_text}/{len(df)} units flagged as text neurons; wrote {out_csv}")
    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--descriptions", type=Path,
                    default=Path("results/descriptions.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/descriptions_annotated.csv"))
    args = ap.parse_args()
    annotate(args.descriptions, args.out)


if __name__ == "__main__":
    main()
