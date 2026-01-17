import argparse
import os
import sys

import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from config import ROOT_DIR
from preprocessing import load_dataset, get_features_and_labels, sample_dataset
from deepL.deep_model import run_transformer_kfold


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a transformer with 5-fold CV.")
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--freeze-layers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-samples", type=float, default=None)
    parser.add_argument("--balance-classes", action="store_true")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(ROOT_DIR, "deep_model_metrics.csv"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_dataset()
    X, y = get_features_and_labels(df)

    X = np.array(X)
    y = np.array(y)

    X, y = sample_dataset(
        X,
        y,
        max_samples=args.max_samples,
        random_seed=args.random_seed,
        balance_classes=args.balance_classes,
    )

    metrics = run_transformer_kfold(
        texts=X,
        labels=y,
        model_name=args.model,
        folds=args.folds,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        freeze_layers=args.freeze_layers,
        learning_rate=args.learning_rate,
        random_seed=args.random_seed,
        output_dir=os.path.join(ROOT_DIR, "metrics", "deep", args.model.replace("/", "_")),
    )

    print(metrics)

    if args.output:
        pd.DataFrame([metrics]).to_csv(args.output, index=False)
        print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
