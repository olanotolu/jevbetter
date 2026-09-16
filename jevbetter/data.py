"""JSONL/CSV loading, n-gram collators, and synthetic data builders."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .featurize import PAD_ID, ngram_ids


@dataclass(frozen=True)
class ChoiceExample:
    context: str
    options: tuple[str, ...]
    label: int


def validate(payload: dict) -> ChoiceExample:
    context, options, label = (
        payload.get("context"),
        payload.get("options"),
        payload.get("label"),
    )
    if not isinstance(context, str) or not isinstance(options, list):
        raise ValueError("each row needs string context and list options")
    if len(options) < 2 or any(
        not isinstance(option, str) or not option for option in options
    ):
        raise ValueError("options must contain at least two non-empty strings")
    if len(set(options)) != len(options):
        raise ValueError("options must not contain duplicates")
    if not isinstance(label, int) or not 0 <= label < len(options):
        raise ValueError("label must be an option index")
    return ChoiceExample(context, tuple(options), label)


class JsonlDataset(Dataset[ChoiceExample]):
    def __init__(self, path: str | Path) -> None:
        with Path(path).open(encoding="utf-8") as handle:
            self.examples = [
                validate(json.loads(line)) for line in handle if line.strip()
            ]
        if not self.examples:
            raise ValueError(f"no examples in {path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class CsvDataset(Dataset[ChoiceExample]):
    """CSV with columns: context, options ("a|b|c"), label (index)."""

    def __init__(self, path: str | Path) -> None:
        with Path(path).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.examples = [
            validate(
                {
                    "context": row["context"],
                    "options": row["options"].split("|"),
                    "label": int(row["label"]),
                }
            )
            for row in rows
        ]
        if not self.examples:
            raise ValueError(f"no examples in {path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def load_dataset(path: str | Path) -> Dataset[ChoiceExample]:
    path = Path(path)
    if path.suffix == ".csv":
        return CsvDataset(path)
    return JsonlDataset(path)


def _tensor_batch(examples, contexts, option_rows, pad_id: int = PAD_ID):
    batch, max_context = len(examples), max(map(len, contexts))
    max_options = max(len(row) for row in option_rows)
    max_option_tokens = max(len(tokens) for row in option_rows for tokens in row)
    context_ids = torch.full((batch, max_context), pad_id, dtype=torch.long)
    option_ids = torch.full(
        (batch, max_options, max_option_tokens), pad_id, dtype=torch.long
    )
    option_mask = torch.zeros((batch, max_options), dtype=torch.bool)
    for row, tokens in enumerate(contexts):
        context_ids[row, : len(tokens)] = torch.tensor(tokens)
    for row, options in enumerate(option_rows):
        option_mask[row, : len(options)] = True
        for column, tokens in enumerate(options):
            option_ids[row, column, : len(tokens)] = torch.tensor(tokens)
    return {
        "context_ids": context_ids,
        "context_mask": context_ids.ne(pad_id),
        "option_ids": option_ids,
        "option_token_mask": option_ids.ne(pad_id),
        "option_mask": option_mask,
        "labels": torch.tensor([item.label for item in examples], dtype=torch.long),
    }


class NgramCollator:
    """Turns (context, options) text into hashed n-gram id batches."""

    def __init__(
        self,
        buckets: int = 65536,
        context_features: int = 512,
        option_features: int = 64,
        ngram_range: tuple[int, int] = (3, 6),
    ) -> None:
        self.buckets = buckets
        self.context_features = context_features
        self.option_features = option_features
        self.ngram_range = ngram_range

    def __call__(self, examples: list[ChoiceExample]):
        contexts = [
            ngram_ids(
                item.context, self.buckets, self.ngram_range, self.context_features
            )
            for item in examples
        ]
        option_rows = [
            [
                ngram_ids(
                    option, self.buckets, self.ngram_range, self.option_features
                )
                for option in item.options
            ]
            for item in examples
        ]
        return _tensor_batch(examples, contexts, option_rows)


class HuggingFaceCollator:
    def __init__(self, tokenizer, context_tokens: int, option_tokens: int) -> None:
        self.tokenizer = tokenizer
        self.context_tokens = context_tokens
        self.option_tokens = option_tokens

    def __call__(self, examples: list[ChoiceExample]):
        contexts = self.tokenizer(
            [item.context for item in examples],
            truncation=True,
            max_length=self.context_tokens,
            add_special_tokens=True,
        )["input_ids"]
        flat = [option for item in examples for option in item.options]
        encoded = self.tokenizer(
            flat,
            truncation=True,
            max_length=self.option_tokens,
            add_special_tokens=True,
        )["input_ids"]
        rows, offset = [], 0
        for item in examples:
            rows.append(encoded[offset : offset + len(item.options)])
            offset += len(item.options)
        return _tensor_batch(
            examples, contexts, rows, self.tokenizer.pad_token_id
        )


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

COLOURS = ("amber", "azure", "bronze", "coral", "crimson", "gold", "green", "indigo")
ANIMALS = ("badger", "crane", "dolphin", "falcon", "gecko", "heron", "ibis", "jaguar")

_HARD_TEMPLATES = (
    "Choose the exact badge {target}. Notes: {notes}. Badge: {target}.",
    "Your target badge is {target}. Ignore the {distractor}. Notes: {notes}.",
    "Pick {target} from the list below. The {distractor} is not the target. Notes: {notes}.",
    "Badge check: find {TARGET}. Notes: {notes}. (Case does not matter.)",
    "Select the badge labeled {target}! Notes: {notes}. Do not pick the {distractor}.",
)


def _near_miss(rng: random.Random, target: str) -> str:
    colour, animal = target.split()
    roll = rng.random()
    if roll < 0.45:
        choices = [c for c in COLOURS if c != colour]
        return f"{rng.choice(choices)} {animal}"
    if roll < 0.75:
        choices = [a for a in ANIMALS if a != animal]
        return f"{colour} {rng.choice(choices)}"
    while True:
        candidate = f"{rng.choice(COLOURS)} {rng.choice(ANIMALS)}"
        if candidate != target:
            return candidate


def synthetic_example(seed: int, mode: str = "hard") -> ChoiceExample:
    rng = random.Random(seed)
    target = f"{rng.choice(COLOURS)} {rng.choice(ANIMALS)}"
    count = rng.randint(2, 8)
    options = {target}
    while len(options) < count:
        options.add(_near_miss(rng, target) if mode == "hard" else f"{rng.choice(COLOURS)} {rng.choice(ANIMALS)}")
    options = list(options)
    rng.shuffle(options)
    notes = " ".join(rng.choice(("north", "south", "east", "west")) for _ in range(8))
    distractor = rng.choice([o for o in options if o != target])
    template = rng.choice(_HARD_TEMPLATES) if mode == "hard" else _HARD_TEMPLATES[0]
    shout = rng.random() < 0.3 and mode == "hard"
    context = template.format(
        target=target,
        TARGET=target.upper() if shout else target,
        distractor=distractor,
        notes=notes,
    )
    return ChoiceExample(context, tuple(options), options.index(target))


def write_synthetic(output: Path, sizes: dict[str, int], seed: int, mode: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    offset = 0
    for split, size in sizes.items():
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for index in range(size):
                item = synthetic_example(seed + offset + index * 104729, mode)
                handle.write(
                    json.dumps(
                        {
                            "context": item.context,
                            "options": item.options,
                            "label": item.label,
                        }
                    )
                    + "\n"
                )
        offset += size * 104729


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--train", type=int, default=4000)
    parser.add_argument("--validation", type=int, default=800)
    parser.add_argument("--test", type=int, default=800)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--mode", choices=("easy", "hard"), default="hard")
    args = parser.parse_args()
    write_synthetic(
        args.output,
        {"train": args.train, "validation": args.validation, "test": args.test},
        args.seed,
        args.mode,
    )


if __name__ == "__main__":
    main()
