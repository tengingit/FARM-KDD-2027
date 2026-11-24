import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import config
from dataloader import MDCDataset, data_loader
from evaluation import evaluate_saved_predictions
from model import FairMDC
from utils.loss import CrossDimPrototypeLoss
from utils.priors import compute_all_priors
from utils.utils import Init_random_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate FARM on predefined benchmark folds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=config.CLASS_EXCLUSIONS, required=True)
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR, help="Dataset directory")
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR, help="Output directory")
    parser.add_argument("--fold", type=int, choices=range(config.NUM_FOLDS),
                        help="Run one zero-based fold. Runs all folds if omitted.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--warmup", type=int, default=config.WARMUP_EPOCHS, help="Number of warm-up epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--device", default=config.DEVICE, help="Compute device, such as cuda, cuda:0, or cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 0 <= args.warmup <= args.epochs:
        parser.error("Use positive epochs and batch size, with warmup between zero and epochs.")
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Install a CUDA-enabled PyTorch build or use --device cpu.")
    return args


def train_fold(dataset, fold, device, args):
    features, labels = dataset.features, dataset.labels
    class_counts = [torch.unique(column).numel() for column in labels.T]
    train_indices, _ = dataset.idx_cv(fold)
    alpha, gamma, conditional = compute_all_priors(labels[train_indices], class_counts)
    refinement = CrossDimPrototypeLoss(
        len(class_counts), class_counts, dim_z=config.EMBEDDING_SIZE,
        momentum=config.PROTOTYPE_MOMENTUM, warmup_epochs=args.warmup,
    )
    refinement.alpha.copy_(alpha)
    refinement.weight_gamma.copy_(gamma)
    refinement.cond_prob.copy_(conditional)
    refinement.to(device)

    model = FairMDC({
        "rand_seed": config.SEED,
        "num_dim": len(class_counts),
        "num_feature": features.shape[1],
        "num_per_dim": class_counts,
        "hidden_size": config.HIDDEN_SIZE,
        "dim_z": config.EMBEDDING_SIZE,
    }).to(device)
    model.reset_parameters()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config.LEARNING_RATE, momentum=config.SGD_MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
    )

    def learning_rate_factor(epoch):
        decay_length = args.epochs - args.warmup
        if epoch < args.warmup or decay_length == 0:
            return 1.0
        return 0.5 * (1 + math.cos(math.pi * (epoch - args.warmup) / decay_length))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    train_loader, test_loader = data_loader(dataset, fold, args.batch_size)
    model.train()
    for epoch in range(args.epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            embeddings, logits = model(x)
            cross_entropy = torch.stack([
                F.cross_entropy(scores, y[:, j]) for j, scores in enumerate(logits)
            ])
            dim_losses = cross_entropy + config.PROTOTYPE_WEIGHT * refinement(embeddings, y, epoch)
            tau = config.LSE_TEMPERATURE
            loss = torch.logsumexp(tau * dim_losses, dim=0) / tau
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    predictions = []
    with torch.no_grad():
        for x, _ in test_loader:
            _, logits = model(x.to(device))
            predictions.append(torch.stack([scores.argmax(dim=1) for scores in logits], dim=1).cpu())
    return torch.cat(predictions).numpy()


def main():
    args = parse_args()
    Init_random_seed(config.SEED)
    dataset = MDCDataset(args.dataset, args.data_dir)
    folds = range(config.NUM_FOLDS) if args.fold is None else [args.fold]
    output = args.output_dir / args.dataset
    output.mkdir(parents=True, exist_ok=True)

    for fold in folds:
        print(f"Training FARM on {args.dataset}, fold {fold}, device {args.device}", flush=True)
        predictions = train_fold(dataset, fold, args.device, args)
        np.savetxt(output / f"fold{fold}_predictions.csv", predictions, fmt="%d", delimiter=",")
    evaluate_saved_predictions(args.dataset, dataset, output, args.fold)


if __name__ == "__main__":
    main()
