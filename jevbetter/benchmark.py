"""Head-to-head benchmark: jevbetter vs the jevlike reference design.

Generates one hard synthetic dataset, trains both models on it with matched
budgets, and prints a markdown comparison table. The jevlike sources are
imported from ``--reference`` (a checkout of vinnylarouge/jevlike); nothing
is copied into this repository.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True,
                        help="checkout of vinnylarouge/jevlike")
    parser.add_argument("--output", type=Path, default=Path("runs/benchmark"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    sys.path.insert(0, str(args.reference))
    from jevlike import data as ref_data
    from jevlike import eval as ref_eval
    from jevlike import model as ref_model
    from jevlike import train as ref_train

    from jevbetter import data as new_data
    from jevbetter import eval as new_eval
    from jevbetter import model as new_model
    from jevbetter import train as new_train

    args.output.mkdir(parents=True, exist_ok=True)
    data_dir = args.output / "data"
    new_data.write_synthetic(
        data_dir,
        {"train": args.train_size, "validation": 800, "test": 800},
        seed=99,
        mode="hard",
    )
    train_path = str(data_dir / "train.jsonl")
    val_path = str(data_dir / "validation.jsonl")
    test_path = str(data_dir / "test.jsonl")

    argv = sys.argv
    try:
        # Reference model, trained exactly the way its README trains it.
        sys.argv = [
            "jevlike-train", train_path,
            "--validation", val_path,
            "--output", str(args.output / "jevlike.pt"),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--device", args.device,
            "--seed", str(args.seed),
        ]
        started = time.perf_counter()
        ref_train.main()
        ref_train_seconds = time.perf_counter() - started

        # jevbetter on the identical files and budget.
        sys.argv = [
            "jevbetter-train", train_path,
            "--validation", val_path,
            "--output", str(args.output / "jevbetter.pt"),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--device", args.device,
            "--seed", str(args.seed),
        ]
        started = time.perf_counter()
        new_train.main()
        new_train_seconds = time.perf_counter() - started
    finally:
        sys.argv = argv

    device = new_model.select_device(args.device)

    ref_checkpoint, ref_collator, _ = ref_model.load_checkpoint(
        args.output / "jevlike.pt", device
    )
    ref_loader = DataLoader(
        ref_data.JsonlDataset(test_path),
        batch_size=args.batch_size,
        collate_fn=ref_collator,
    )
    started = time.perf_counter()
    ref_metrics = ref_eval.metrics(ref_checkpoint, ref_loader, device)
    ref_seconds = time.perf_counter() - started

    new_checkpoint, new_collator, _, temperature = new_model.load_checkpoint(
        args.output / "jevbetter.pt", device
    )
    new_loader = DataLoader(
        new_data.JsonlDataset(test_path),
        batch_size=args.batch_size,
        collate_fn=new_collator,
    )
    started = time.perf_counter()
    new_metrics = new_eval.metrics(
        new_checkpoint, new_loader, device, temperature
    )
    new_seconds = time.perf_counter() - started

    ref_metrics["menus_per_second"] = (
        ref_metrics["examples"] / ref_seconds
    )
    row = {
        "model": ["jevlike (reference)", "jevbetter"],
        "top1": [ref_metrics["top1"], new_metrics["top1"]],
        "top3": [ref_metrics["top3"], new_metrics["top3"]],
        "mrr": ["n/a", new_metrics["mrr"]],
        "ece": [ref_metrics["ece"], new_metrics["ece"]],
        "menus_per_second": [
            ref_metrics["menus_per_second"], new_metrics["menus_per_second"]
        ],
        "train_seconds": [ref_train_seconds, new_train_seconds],
    }
    print(f"test examples: {new_metrics['examples']}, device: {device}")
    print("| model | top-1 | top-3 | MRR | ECE ↓ | menus/sec | train sec |")
    print("|---|---|---|---|---|---|---|")
    for i in range(2):
        mrr = row["mrr"][i]
        mrr = f"{mrr:.3f}" if isinstance(mrr, float) else mrr
        print(
            f"| {row['model'][i]} | {row['top1'][i]:.3f} | {row['top3'][i]:.3f} | "
            f"{mrr} | {row['ece'][i]:.4f} | {row['menus_per_second'][i]:.0f} | "
            f"{row['train_seconds'][i]:.0f} |"
        )
    (args.output / "results.json").write_text(
        json.dumps(
            {
                "config": {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "train_size": args.train_size,
                    "seed": args.seed,
                    "device": str(device),
                },
                "jevlike": {k: v[0] for k, v in row.items() if k != "model"},
                "jevbetter": {k: v[1] for k, v in row.items() if k != "model"},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
