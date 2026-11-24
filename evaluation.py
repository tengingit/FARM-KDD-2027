import argparse
import csv
from pathlib import Path

import numpy as np

import config
from dataloader import MDCDataset


def evaluate(targets, predictions, exclusions):
    if predictions.shape != targets.shape:
        raise ValueError(f"Expected {targets.shape} predictions, got {predictions.shape}.")
    correct = targets == predictions
    balanced_accuracies = []
    worst_recalls = []
    for dim in range(targets.shape[1]):
        classes = np.union1d(targets[:, dim], predictions[:, dim])
        recalls = {}
        for label in classes:
            members = targets[:, dim] == label
            recalls[label] = float(correct[members, dim].mean()) if members.any() else 0.0
        balanced_accuracies.append(float(np.mean(list(recalls.values()))))
        worst_recalls.append(min(
            recall for label, recall in recalls.items() if label not in exclusions.get(dim, ())
        ))
    return {
        "HammingScore": float(correct.mean()),
        "ExactMatch": float(correct.all(axis=1).mean()),
        "SubExactMatch": float((correct.sum(axis=1) >= targets.shape[1] - 1).mean()),
        "WDBAcc": min(balanced_accuracies),
        "MinWCRec": min(worst_recalls),
        **{f"WCRec_dim{dim + 1}": recall for dim, recall in enumerate(worst_recalls)},
    }


def evaluate_saved_predictions(dataset_name, dataset, prediction_dir, fold=None):
    folds = range(config.NUM_FOLDS) if fold is None else [fold]
    rows = []
    for fold_index in folds:
        prediction_file = prediction_dir / f"fold{fold_index}_predictions.csv"
        predictions = np.loadtxt(prediction_file, delimiter=",", dtype=int, ndmin=2)
        _, test_indices = dataset.idx_cv(fold_index)
        targets = dataset.labels[test_indices].numpy()
        metrics = evaluate(targets, predictions, config.CLASS_EXCLUSIONS[dataset_name])
        rows.append({"Fold": fold_index, **metrics})

    filename = "metrics.csv" if fold is None else f"fold{fold}_metrics.csv"
    with (prediction_dir / filename).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"{dataset_name}: mean +/- standard deviation across {len(rows)} fold(s)")
    for metric in metrics:
        scores = [row[metric] for row in rows]
        print(f"  {metric}: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Compute FARM metrics from saved test predictions.")
    parser.add_argument("--dataset", choices=config.CLASS_EXCLUSIONS, required=True)
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR, help="Dataset directory")
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR,
                        help="Directory containing per-dataset prediction files")
    parser.add_argument("--fold", type=int, choices=range(config.NUM_FOLDS),
                        help="Evaluate one zero-based fold. Evaluates all folds if omitted.")
    args = parser.parse_args()
    dataset = MDCDataset(args.dataset, args.data_dir)
    evaluate_saved_predictions(args.dataset, dataset, args.output_dir / args.dataset, args.fold)


if __name__ == "__main__":
    main()
