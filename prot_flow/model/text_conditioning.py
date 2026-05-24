from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


@dataclass
class TextConditioningOutput:
    token_condition: torch.Tensor
    global_condition: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None


class GlobalConditionProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, pooled_text_embedding: torch.Tensor) -> torch.Tensor:
        return self.net(pooled_text_embedding)


class TokenConditionProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(token_embeddings)


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, hidden_size: int, condition_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.to_scale = nn.Linear(condition_dim, hidden_size)
        self.to_shift = nn.Linear(condition_dim, hidden_size)

    def forward(self, hidden_states: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if condition.dim() == 2:
            condition = condition[:, None, :]
        mean = hidden_states.mean(dim=-1, keepdim=True)
        var = hidden_states.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (hidden_states - mean) / torch.sqrt(var + self.eps)
        scale = 1 + self.to_scale(condition)
        shift = self.to_shift(condition)
        return normalized * scale + shift


class NullConditioning(nn.Module):
    def __init__(self, condition_dim: int):
        super().__init__()
        self.null_global = nn.Parameter(torch.zeros(1, condition_dim))

    def forward(self, batch_size: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> TextConditioningOutput:
        token_condition = torch.zeros(batch_size, seq_len, self.null_global.shape[-1], device=device, dtype=dtype)
        global_condition = self.null_global.expand(batch_size, -1).to(device=device, dtype=dtype)
        attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)
        return TextConditioningOutput(
            token_condition=token_condition,
            global_condition=global_condition,
            attention_mask=attention_mask,
        )


class TextConditionEncoder(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        condition_dim: int,
        max_length: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size
        self.max_length = max_length
        self.token_projector = TokenConditionProjector(hidden_size, condition_dim, dropout=dropout)
        self.global_projector = GlobalConditionProjector(hidden_size, condition_dim, dropout=dropout)

    def forward(self, batch: dict | Sequence[str], device: torch.device) -> TextConditioningOutput:
        if isinstance(batch, dict):
            texts = batch.get("texts", batch.get("text"))
        else:
            texts = batch
        if texts is None:
            raise ValueError("TextConditionEncoder expected `texts`/`text` in batch or a sequence of strings.")
        if isinstance(texts, str):
            texts = [texts]

        tokenized = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        outputs = self.encoder(**tokenized)
        hidden = outputs.last_hidden_state
        pooled = hidden[:, 0]
        return TextConditioningOutput(
            token_condition=self.token_projector(hidden),
            global_condition=self.global_projector(pooled),
            attention_mask=tokenized["attention_mask"],
        )
