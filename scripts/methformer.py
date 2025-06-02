import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import Dataset
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput, SequenceClassifierOutput


class MethformerDataset(Dataset):
    """
    Unified dataset for Methformer:
    - 'pretrain' mode: returns masked input chunks and original values for reconstruction
    - 'regression' mode: returns full region input and scalar label
    """

    def __init__(
        self,
        inputs,
        labels=None,
        mode="pretrain",
        chunk_size=128,
        mask_value=-1.0,
        masking_ratio=0.15,
    ):
        """
        inputs:
            - pretrain mode: Tensor (N, R, C)
            - regression mode: Tensor (N, L, C)
        labels:
            - pretrain mode: None
            - regression mode: Tensor (N,)
        """
        assert mode in {"pretrain", "regression"}
        self.mode = mode
        self.mask_value = mask_value
        self.masking_ratio = masking_ratio

        if mode == "pretrain":
            self.data = inputs
            self.n_samples, self.n_regions, self.n_channels = self.data.shape
            self.chunk_size = min(chunk_size, self.n_regions)
        else:  # regression
            assert labels is not None
            assert inputs.shape[0] == labels.shape[0]
            self.inputs = torch.tensor(inputs, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        if self.mode == "pretrain":
            return self.n_samples * (self.n_regions // self.chunk_size)
        else:
            return self.inputs.shape[0]

    def __getitem__(self, idx):
        if self.mode == "pretrain":
            sample_idx = idx % self.n_samples
            chunk_start = random.randint(0, self.n_regions - self.chunk_size)
            chunk = self.data[
                sample_idx, chunk_start : chunk_start + self.chunk_size, :
            ]

            x = torch.tensor(chunk, dtype=torch.float32)
            mask = torch.rand(self.chunk_size) < self.masking_ratio
            x_masked = x.clone()
            x_masked[mask] = self.mask_value

            return {
                "inputs": x_masked,
                "labels": x,
                "attention_mask": ~mask,
            }

        else:  # regression
            x = self.inputs[idx]  # (L, C)
            y = self.labels[idx]  # scalar
            attention_mask = (x != self.mask_value).any(dim=-1).long()
            return {
                "input_values": x,
                "labels": y,
                "attention_mask": attention_mask,
            }


class MethformerCollator:
    """
    Collator for Methformer dataset.
    Handles both pretraining and regression modes.
    - Pretrain: returns masked inputs and original labels
    - Regression: returns full input and scalar label
    """
    def __init__(self, mode="pretrain"):
        assert mode in {"pretrain", "regression"}
        self.mode = mode

    def __call__(self, batch):
        def ensure_tensor(x, dtype=torch.float32):
            return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=dtype)

        if self.mode == "pretrain":
            inputs = [ensure_tensor(item["inputs"]) for item in batch]
            labels = [ensure_tensor(item["labels"]) for item in batch]
            attention_mask = [
                ensure_tensor(item["attention_mask"], dtype=torch.bool)
                for item in batch
            ]

            return {
                "input_values": torch.stack(inputs),
                "labels": torch.stack(labels),
                "attention_mask": torch.stack(attention_mask),
            }

        else:  # regression
            inputs = [ensure_tensor(item["input_values"]) for item in batch]
            labels = [ensure_tensor(item["labels"]) for item in batch]
            attention_mask = [
                ensure_tensor(item["attention_mask"], dtype=torch.bool)
                for item in batch
            ]

            return {
                "input_values": torch.stack(inputs),
                "labels": torch.stack(labels),
                "attention_mask": torch.stack(attention_mask),
            }


class Methformer(PreTrainedModel):
    """
    Methformer model for both:
    - Pretraining (signal → signal, masked reconstruction)
    - Fine-tuning (signal → scalar, regression)
    """

    def __init__(self, config, mode="pretrain", use_cls_token=False):
        super().__init__(config)
        self.config = config
        if mode not in {"pretrain", "regression"}:
            raise ValueError(
                f"Unsupported mode '{self.mode}'. Must be 'pretrain' or 'regression'."
            )

        self.mode = mode
        self.use_cls_token = use_cls_token

        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        self.pos_embed = nn.Parameter(
            torch.randn(
                1,
                config.max_position_embeddings + int(use_cls_token),
                config.hidden_dim,
            )
        )

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_attention_heads,
            dropout=config.hidden_dropout_prob,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_hidden_layers
        )

        self.pretrain_head = nn.Linear(config.hidden_dim, config.input_dim)
        self.regression_head = nn.Linear(config.hidden_dim, 1)

    def forward(self, input_values, attention_mask, labels=None):
        B, L, _ = input_values.shape

        x = self.input_proj(input_values)  # (B, L, D)

        if self.use_cls_token:
            cls_token = self.cls_token.expand(B, 1, -1)  # (B, 1, D)
            x = torch.cat([cls_token, x], dim=1)  # (B, L+1, D)
            attention_mask = torch.cat(
                [
                    torch.ones(
                        B, 1, dtype=attention_mask.dtype, device=attention_mask.device
                    ),
                    attention_mask,
                ],
                dim=1,
            )

        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.encoder(
            x, src_key_padding_mask=~attention_mask.bool()
        )  # (B, L(+1), D)

        if self.mode == "pretrain":
            output = self.pretrain_head(
                x[:, 1:] if self.use_cls_token else x
            )  # ignore CLS if present
            loss = None
            if labels is not None:
                mask = attention_mask[:, 1:] if self.use_cls_token else attention_mask
                mask = mask.unsqueeze(-1).expand_as(labels)
                loss = F.mse_loss(output[mask], labels[mask])
            return ModelOutput(loss=loss, last_hidden_state=output)

        elif self.mode == "regression":
            if self.use_cls_token:
                pooled = x[:, 0]  # CLS token
            else:
                pooled = (x * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
                    1, keepdim=True
                ).clamp(min=1)
            logits = self.regression_head(pooled).squeeze(-1)
            loss = F.mse_loss(logits, labels) if labels is not None else None
            return SequenceClassifierOutput(loss=loss, logits=logits)

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def set_mode(self, mode: str):
        assert mode in {"pretrain", "regression"}, f"Invalid mode: {mode}"
        self.mode = mode

    @classmethod
    def from_pretrained_encoder(
        cls, pretrained_path, mode="regression", use_cls_token=False
    ):
        base_model = cls.from_pretrained(
            pretrained_path, mode="pretrain", use_cls_token=use_cls_token
        )
        new_model = cls(base_model.config, mode=mode, use_cls_token=use_cls_token)
        new_model.encoder.load_state_dict(base_model.encoder.state_dict())
        return new_model

    def predict(self, dataloader, device="cuda"):
        self.eval()
        self.to(device)

        all_logits = []
        all_labels = []

        for batch in dataloader:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.no_grad():
                outputs = self(input_values=input_values, attention_mask=attention_mask)
                logits = outputs.logits

            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        y_pred = np.concatenate(all_logits)
        y_true = np.concatenate(all_labels)

        metrics = {
            "mse": mean_squared_error(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }

        return y_pred, y_true, metrics
