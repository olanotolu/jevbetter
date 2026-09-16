"""Train a scorer: cosine schedule, early stopping, temperature scaling."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import load_dataset
from .model import make_system, select_device, trainable_state


def move(batch, device):
    return {name: tensor.to(device) for name, tensor in batch.items()}


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        batch = move(batch, device)
        all_logits.append(model(batch).cpu())
        all_labels.append(batch["labels"].cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


def mean_loss(model, loader, device, label_smoothing=0.0):
    logits, labels = collect_logits(model, loader, device)
    return float(
        F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
    )


def fit_temperature(model, loader, device) -> float:
    """Pick the softmax temperature that minimises validation NLL."""
    logits, labels = collect_logits(model, loader, device)
    best_temperature, best_nll = 1.0, float("inf")
    candidates = torch.logspace(math.log10(0.05), math.log10(20.0), 80)
    for temperature in candidates.tolist():
        nll = float(F.cross_entropy(logits / temperature, labels))
        if nll < best_nll:
            best_temperature, best_nll = temperature, nll
    # Local refinement around the grid winner.
    for _ in range(3):
        step = best_temperature * 0.1
        for temperature in (
            best_temperature - step,
            best_temperature + step,
        ):
            if temperature <= 0:
                continue
            nll = float(F.cross_entropy(logits / temperature, labels))
            if nll < best_nll:
                best_temperature, best_nll = temperature, nll
    return best_temperature


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train")
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", default="runs/model.pt")
    parser.add_argument("--encoder", choices=("ngram", "hf"), default="ngram")
    parser.add_argument("--hf-model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--buckets", type=int, default=65536)
    parser.add_argument("--context-features", type=int, default=512)
    parser.add_argument("--option-features", type=int, default=64)
    parser.add_argument("--context-tokens", type=int, default=192)
    parser.add_argument("--option-tokens", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--warmup", type=float, default=0.05)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = select_device(args.device)
    config = {
        "encoder": args.encoder,
        "hf_model": args.hf_model,
        "width": args.width,
        "heads": args.heads,
        "layers": args.layers,
        "buckets": args.buckets,
        "context_features": args.context_features,
        "option_features": args.option_features,
        "context_tokens": args.context_tokens,
        "option_tokens": args.option_tokens,
    }
    model, collator = make_system(config, device)
    train_loader = DataLoader(
        load_dataset(args.train),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        load_dataset(args.validation),
        batch_size=args.batch_size,
        collate_fn=collator,
    )
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimiser = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = max(1, int(total_steps * args.warmup))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, schedule)

    best_loss, best_state, stale, step = float("inf"), None, 0, 0
    for epoch in range(args.epochs):
        model.train()
        total, count = 0.0, 0
        for host_batch in train_loader:
            batch = move(host_batch, device)
            loss = F.cross_entropy(
                model(batch),
                batch["labels"],
                label_smoothing=args.label_smoothing,
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimiser.step()
            scheduler.step()
            step += 1
            total += float(loss.detach()) * batch["labels"].numel()
            count += batch["labels"].numel()
        validation_loss = mean_loss(model, validation_loader, device)
        improved = validation_loss < best_loss - 1e-4
        if improved:
            best_loss, best_state, stale = validation_loss, trainable_state(model), 0
        else:
            stale += 1
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "train_nll": total / count,
                    "validation_nll": validation_loss,
                    "lr": scheduler.get_last_lr()[0],
                    "device": str(device),
                }
            )
        )
        if stale >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch + 1}))
            break

    model.load_state_dict(best_state, strict=False)
    temperature = fit_temperature(model, validation_loader, device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"config": config, "state_dict": best_state, "temperature": temperature},
        output,
    )
    print(
        json.dumps(
            {
                "checkpoint": str(output),
                "best_validation_nll": best_loss,
                "temperature": temperature,
            }
        )
    )


if __name__ == "__main__":
    main()
