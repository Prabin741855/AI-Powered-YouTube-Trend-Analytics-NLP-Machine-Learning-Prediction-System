from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from youtube_analytics.config import DATA_PATH, MODEL_PATH
from youtube_analytics.data import load_dataset
from youtube_analytics.database import save_model_metrics
from youtube_analytics.modeling import train_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Train time-aware YouTube trend and performance models.")
    parser.add_argument("--input", type=Path, default=DATA_PATH, help="Input YouTube videos CSV")
    parser.add_argument("--output", type=Path, default=MODEL_PATH, help="Serialized model bundle destination")
    parser.add_argument("--metrics-csv", type=Path, help="Optional path to export holdout metrics")
    parser.add_argument("--no-xgboost", action="store_true", help="Train only the scikit-learn models")
    args = parser.parse_args()
    frame, report = load_dataset(args.input)
    print(f"Training on {len(frame):,} cleaned videos from {report['date_min']} to {report['date_max']}.")
    bundle, metrics = train_models(frame, args.output, include_xgboost=not args.no_xgboost)
    save_model_metrics(metrics)
    if args.metrics_csv:
        args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(args.metrics_csv, index=False)
    print(f"Time-aware holdout begins {bundle['cutoff']}; model bundle saved to {args.output}.")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()