"""Evaluate accuracy, ranking, calibration, throughput and a control."""

from __future__ import annotations

import argparse
import json
import time

import torch
from torch.utils.data import DataLoader

from .data import load_dataset
from .model import load_checkpoint, select_device
from .train import move


@torch.no_grad()
def metrics(model, loader, device, temperature=1.0, shuffle_context=False):
    model.eval()
    top1 = top3 = top5 = total = 0
    reciprocal_rank = 0.0
    confidences, predictions, labels, sizes = [], [], [], []
    started = time.perf_counter()
    for host_batch in loader:
        batch = move(host_batch, device)
        logits = (model(batch, shuffle_context=shuffle_context) / temperature).cpu()
        n_real = batch["option_mask"].sum(1).cpu()
        order = logits.argsort(-1, descending=True)
        batch_labels = batch["labels"].cpu()
        arange = torch.arange(order.shape[1])[None, :]
        hits = (order == batch_labels[:, None]) & (arange < n_real[:, None])
        top1 += int(hits[:, :1].any(1).sum())
        top3 += int(hits[:, :3].any(1).sum())
        top5 += int(hits[:, :5].any(1).sum())
        rank = hits.float().argmax(1) + 1
        reciprocal_rank += float((1.0 / rank).sum())
        total += batch_labels.numel()
        probabilities = (logits).softmax(-1)
        confidence, prediction = probabilities.max(-1)
        confidences.append(confidence)
        predictions.append(prediction)
        labels.append(batch_labels)
        sizes.append(n_real)
    elapsed = time.perf_counter() - started
    confidence = torch.cat(confidences)
    prediction = torch.cat(predictions)
    labels = torch.cat(labels)
    ece = 0.0
    for lower in torch.linspace(0, 0.9, 10):
        selected = (confidence >= lower) & (confidence < lower + 0.1)
        if selected.any():
            gap = prediction[selected].eq(labels[selected]).float().mean()
            gap -= confidence[selected].mean()
            ece += float(selected.float().mean() * gap.abs())
    sizes = torch.cat(sizes)
    per_size = {}
    for size in sorted(set(sizes.tolist())):
        selected = sizes == size
        per_size[str(size)] = float(
            prediction[selected].eq(labels[selected]).float().mean()
        )
    return {
        "top1": top1 / total,
        "top3": top3 / total,
        "top5": top5 / total,
        "mrr": reciprocal_rank / total,
        "ece": ece,
        "examples": total,
        "menus_per_second": total / elapsed,
        "accuracy_by_menu_size": per_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    device = select_device(args.device)
    model, collator, _, temperature = load_checkpoint(args.checkpoint, device)
    loader = DataLoader(
        load_dataset(args.data), batch_size=args.batch_size, collate_fn=collator
    )
    print(
        json.dumps(
            {
                "model": metrics(model, loader, device, temperature),
                "temperature": temperature,
                "shuffled_context": metrics(
                    model, loader, device, temperature, shuffle_context=True
                ),
                "device": str(device),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
