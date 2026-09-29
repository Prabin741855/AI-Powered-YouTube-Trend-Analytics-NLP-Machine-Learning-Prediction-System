from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from youtube_analytics.config import DATABASE_PATH, DATA_PATH, LOGGER
from youtube_analytics.data import load_dataset
from youtube_analytics.database import persist_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean INvideos.csv and persist the validated rows to SQLite.")
    parser.add_argument("--input", type=Path, default=DATA_PATH, help="Input YouTube videos CSV")
    parser.add_argument("--database", type=Path, default=DATABASE_PATH, help="SQLite database destination")
    parser.add_argument("--cleaned-csv", type=Path, help="Optional destination for the cleaned feature dataset")
    args = parser.parse_args()
    frame, report = load_dataset(args.input)
    persist_dataset(frame, args.database)
    if args.cleaned_csv:
        args.cleaned_csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.cleaned_csv, index=False)
    print(json.dumps(report, indent=2))
    LOGGER.info("Prepared %s rows from %s", len(frame), args.input)


if __name__ == "__main__":
    main()