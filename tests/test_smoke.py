import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from jevbetter.data import NgramCollator, load_dataset, synthetic_example
from jevbetter.featurize import char_ngrams, ngram_ids
from jevbetter.model import NgramScorer, load_checkpoint
from jevbetter.train import fit_temperature, move


def test_ngrams_are_stable_case_insensitive_and_padded():
    first = ngram_ids("Amber Badger", 1024)
    second = ngram_ids("amber badger", 1024)
    assert first == second
    assert all(1 <= i < 1024 for i in first)
    assert char_ngrams("") == []


def test_tiny_scorer_learns_and_normalises(tmp_path):
    torch.manual_seed(5)
    examples = [synthetic_example(1000 + index, "hard") for index in range(48)]
    collator = NgramCollator(buckets=4096, context_features=256, option_features=32)
    loader = DataLoader(examples, batch_size=16, collate_fn=collator)
    batch = next(iter(loader))
    model = NgramScorer(width=32, buckets=4096, heads=2, layers=1,
                        context_features=256)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    initial = float(F.cross_entropy(model(batch), batch["labels"]).detach())
    for _ in range(40):
        loss = F.cross_entropy(model(batch), batch["labels"])
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    logits = model(batch)
    final = float(F.cross_entropy(logits, batch["labels"]).detach())
    assert final < initial * 0.6
    assert torch.allclose(
        logits.softmax(-1).sum(-1), torch.ones(len(batch["labels"])), atol=1e-5
    )


def test_checkpoint_round_trip_with_temperature(tmp_path):
    torch.manual_seed(11)
    examples = [synthetic_example(2000 + index, "hard") for index in range(24)]
    collator = NgramCollator(buckets=2048, context_features=128, option_features=24)
    batch = collator(examples)
    model = NgramScorer(width=24, buckets=2048, heads=2, layers=1,
                        context_features=128)
    temperature = fit_temperature(
        model,
        DataLoader(examples, batch_size=24, collate_fn=collator),
        torch.device("cpu"),
    )
    assert 0.05 <= temperature <= 20.0
    from jevbetter.model import trainable_state

    path = tmp_path / "model.pt"
    torch.save(
        {
            "config": {
                "encoder": "ngram", "hf_model": "x", "width": 24, "heads": 2,
                "layers": 1, "buckets": 2048, "context_features": 128,
                "option_features": 24, "context_tokens": 192,
                "option_tokens": 32,
            },
            "state_dict": trainable_state(model),
            "temperature": temperature,
        },
        path,
    )
    loaded, _, _, loaded_temperature = load_checkpoint(path, torch.device("cpu"))
    assert loaded_temperature == temperature
    with torch.no_grad():
        before = (model(batch) / temperature).softmax(-1)
        after = (loaded(move(batch, torch.device("cpu"))) / loaded_temperature).softmax(-1)
    assert torch.allclose(before, after, atol=1e-5)


def test_dataset_loaders_reject_bad_rows(tmp_path):
    import json

    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"context": "x", "options": ["a"], "label": 0}) + "\n")
    try:
        load_dataset(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a single-option row")
