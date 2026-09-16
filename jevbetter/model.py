"""One-pass option scorers with a hashed n-gram encoder and a contrastive head.

Where the reference design scores each option independently with a single
dot product, this model

1. encodes the context with a tiny transformer over hashed character
   n-grams (subword features, still CPU-cheap);
2. lets options attend to *each other* before scoring, so near-miss rivals
   sharpen the decision instead of confusing it;
3. replaces the raw dot product with a gated two-layer scorer.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from .featurize import PAD_ID


class MultiHeadCrossAttention(nn.Module):
    """Each option queries the context with several attention heads."""

    def __init__(self, width: int, heads: int = 4) -> None:
        super().__init__()
        assert width % heads == 0, "width must divide evenly into heads"
        self.heads = heads
        self.head_dim = width // heads
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def forward(
        self,
        options: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, n_options, _ = options.shape
        n_context = context.shape[1]

        def split(tensor):
            return (
                tensor.view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
            )

        query, key, value = (
            split(self.query(options)),
            split(self.key(context)),
            split(self.value(context)),
        )
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            ~context_mask[:, None, None, :], torch.finfo(scores.dtype).min
        )
        attended = torch.matmul(scores.softmax(-1), value)
        merged = attended.transpose(1, 2).reshape(batch, n_options, -1)
        return self.out(merged)


class OptionMixer(nn.Module):
    """Self-attention over the option set: each option sees its rivals."""

    def __init__(self, width: int, heads: int = 4) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            width, heads, batch_first=True, dropout=0.0
        )
        self.norm = nn.LayerNorm(width)

    def forward(
        self, options: torch.Tensor, option_mask: torch.Tensor
    ) -> torch.Tensor:
        mixed, _ = self.attention(
            options,
            options,
            options,
            key_padding_mask=~option_mask,
            need_weights=False,
        )
        return self.norm(options + mixed)


class GatedScorerHead(nn.Module):
    """Contrastive head: rival-aware options, multi-head context attention,
    and a gated MLP instead of a single dot product."""

    def __init__(self, width: int, heads: int = 4) -> None:
        super().__init__()
        self.option_norm = nn.LayerNorm(width)
        self.context_norm = nn.LayerNorm(width)
        self.mixer = OptionMixer(width, heads)
        self.cross = MultiHeadCrossAttention(width, heads)
        self.gate = nn.Linear(width, width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1)
        )

    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        options: torch.Tensor,
        option_mask: torch.Tensor,
        shuffle_context: bool = False,
    ) -> torch.Tensor:
        context = self.context_norm(context.float())
        options = self.option_norm(options.float())
        if shuffle_context and context.shape[0] > 1:
            context = context.roll(1, dims=0)
            context_mask = context_mask.roll(1, dims=0)
        rival_aware = self.mixer(options, option_mask)
        attended = self.cross(rival_aware, context, context_mask)
        interaction = rival_aware * attended
        gated = interaction * torch.sigmoid(self.gate(interaction))
        logits = self.mlp(gated).squeeze(-1)
        return logits.masked_fill(~option_mask, torch.finfo(logits.dtype).min)


class TinyContextEncoder(nn.Module):
    """A two-layer transformer over n-gram embeddings: local order plus
    long-range context mixing, still tiny enough for a laptop CPU."""

    def __init__(self, width: int, layers: int = 2, heads: int = 4) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            width,
            heads,
            dim_feedforward=width * 2,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, layers)

    def forward(
        self, embeddings: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        return self.encoder(embeddings, src_key_padding_mask=~mask)


class NgramScorer(nn.Module):
    def __init__(
        self,
        width: int = 128,
        buckets: int = 65536,
        heads: int = 4,
        layers: int = 2,
        context_features: int = 512,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(buckets, width, padding_idx=PAD_ID)
        self.position = nn.Embedding(context_features, width)
        self.context_encoder = TinyContextEncoder(width, layers, heads)
        self.head = GatedScorerHead(width, heads)

    def forward(self, batch: dict[str, torch.Tensor], shuffle_context: bool = False):
        context_ids = batch["context_ids"]
        positions = torch.arange(context_ids.shape[1], device=context_ids.device)
        context = self.embedding(context_ids) + self.position(positions)
        context = self.context_encoder(context, batch["context_mask"])
        option_tokens = self.embedding(batch["option_ids"])
        weights = batch["option_token_mask"].unsqueeze(-1).float()
        options = (option_tokens * weights).sum(2) / weights.sum(2).clamp_min(1)
        return self.head(
            context,
            batch["context_mask"],
            options,
            batch["option_mask"],
            shuffle_context,
        )


class FrozenTransformerScorer(nn.Module):
    """Same contrastive head on top of a frozen Hugging Face encoder."""

    def __init__(self, model_name: str, width: int = 128, heads: int = 4) -> None:
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.requires_grad_(False).eval()
        hidden = self.encoder.config.hidden_size
        self.project = (
            nn.Linear(hidden, width) if hidden != width else nn.Identity()
        )
        self.head = GatedScorerHead(width, heads)

    def forward(self, batch: dict[str, torch.Tensor], shuffle_context: bool = False):
        self.encoder.eval()
        with torch.no_grad():
            context = self.project(
                self.encoder(
                    input_ids=batch["context_ids"],
                    attention_mask=batch["context_mask"],
                ).last_hidden_state
            )
            shape = batch["option_ids"].shape
            flat_ids = batch["option_ids"].reshape(-1, shape[-1])
            flat_mask = batch["option_token_mask"].reshape(-1, shape[-1])
            hidden = self.project(
                self.encoder(input_ids=flat_ids, attention_mask=flat_mask)
                .last_hidden_state
            )
            pooled = (hidden * flat_mask.unsqueeze(-1)).sum(1)
            pooled = pooled / flat_mask.sum(1, keepdim=True).clamp_min(1)
            options = pooled.reshape(shape[0], shape[1], -1)
        return self.head(
            context,
            batch["context_mask"],
            options,
            batch["option_mask"],
            shuffle_context,
        )


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_system(config: dict, device: torch.device):
    from .data import HuggingFaceCollator, NgramCollator

    if config["encoder"] == "ngram":
        model = NgramScorer(
            width=config["width"],
            buckets=config["buckets"],
            heads=config["heads"],
            layers=config["layers"],
            context_features=config["context_features"],
        )
        collator = NgramCollator(
            buckets=config["buckets"],
            context_features=config["context_features"],
            option_features=config["option_features"],
        )
    else:
        from transformers import AutoTokenizer

        name = config["hf_model"]
        tokenizer = AutoTokenizer.from_pretrained(name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = FrozenTransformerScorer(name, config["width"], config["heads"])
        collator = HuggingFaceCollator(
            tokenizer, config["context_tokens"], config["option_tokens"]
        )
    return model.to(device), collator


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def load_checkpoint(path: str | Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model, collator = make_system(payload["config"], device)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if unexpected or any(not name.startswith("encoder.") for name in missing):
        raise ValueError(
            f"checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    return model, collator, payload["config"], float(payload.get("temperature", 1.0))
