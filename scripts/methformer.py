import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import Dataset
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput, SequenceClassifierOutput


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean", pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer(
            "pos_weight", torch.tensor(pos_weight) if pos_weight is not None else None
        )

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none", pos_weight=self.pos_weight
        )
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_term = (1 - pt) ** self.gamma
        loss = self.alpha * focal_term * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class Methformer(PreTrainedModel):
    def __init__(self, config, mode="pretrain", use_cls_token=False, pos_weight=None):
        super().__init__(config)
        self.config = config
        self.mode = mode
        self.use_cls_token = use_cls_token
        self.pos_weight = pos_weight if mode == "binary_classification" else None

        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)

        if mode not in {"pretrain", "regression", "binary_classification"}:
            raise ValueError(
                f"Unsupported mode '{mode}'. Must be 'pretrain', 'regression', or 'binary_classification'."
            )

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
        self.regression_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.binary_classification_head = nn.Sequential(nn.Linear(config.hidden_dim, 1))

        if mode == "binary_classification":
            self.loss_fn = FocalLoss(
                alpha=1.0, gamma=2.0, reduction="mean", pos_weight=pos_weight
            )
        else:
            self.loss_fn = None

    def forward(self, input_values, attention_mask, labels=None):
        B, L, _ = input_values.shape
        x = self.input_proj(input_values)

        if self.use_cls_token:
            cls_token = self.cls_token.expand(B, 1, -1)
            x = torch.cat([cls_token, x], dim=1)
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
        x = self.encoder(x, src_key_padding_mask=~attention_mask.bool())

        if self.mode == "pretrain":
            output = self.pretrain_head(x[:, 1:] if self.use_cls_token else x)
            loss = None
            if labels is not None:
                mask = attention_mask[:, 1:] if self.use_cls_token else attention_mask
                mask = mask.unsqueeze(-1).expand_as(labels)
                loss = F.mse_loss(output[mask], labels[mask])
            return ModelOutput(loss=loss, last_hidden_state=output)

        elif self.mode == "regression":
            pooled = (
                x[:, 0]
                if self.use_cls_token
                else (x * attention_mask.unsqueeze(-1)).sum(1)
                / attention_mask.sum(1, keepdim=True).clamp(min=1)
            )
            logits = self.regression_head(pooled).squeeze(-1)
            loss = F.mse_loss(logits, labels) if labels is not None else None
            return SequenceClassifierOutput(loss=loss, logits=logits)

        elif self.mode == "binary_classification":
            pooled = (
                x[:, 0]
                if self.use_cls_token
                else (x * attention_mask.unsqueeze(-1)).sum(1)
                / attention_mask.sum(1, keepdim=True).clamp(min=1)
            )
            logits = self.binary_classification_head(pooled).squeeze(-1)
            loss = self.loss_fn(logits, labels) if labels is not None else None
            return SequenceClassifierOutput(loss=loss, logits=logits)

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def set_mode(self, mode: str):
        assert mode in {"pretrain", "regression", "binary_classification"}, (
            f"Invalid mode: {mode}"
        )
        self.mode = mode

    @classmethod
    def from_pretrained_encoder(
        cls, path, mode="pretrain", use_cls_token=False, pos_weight=None
    ):
        config = PretrainedConfig.from_pretrained(path)
        model = cls(
            config, mode=mode, use_cls_token=use_cls_token, pos_weight=pos_weight
        )

        state_dict = load_file(os.path.join(path, "model.safetensors"))
        pretrained = cls(config, mode="pretrain", use_cls_token=use_cls_token)
        pretrained.load_state_dict(state_dict, strict=False)
        model.encoder.load_state_dict(pretrained.encoder.state_dict())

        return model

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


class MethformerDataset(Dataset):
    def __init__(
        self,
        inputs,
        labels=None,
        mode="pretrain",
        chunk_size=128,
        mask_value=-1.0,
        masking_ratio=0.15,
    ):
        self.mode = mode
        self.mask_value = mask_value
        self.masking_ratio = masking_ratio

        if mode == "pretrain":
            self.data = inputs
            self.n_samples, self.n_regions, self.n_channels = self.data.shape
            self.chunk_size = min(chunk_size, self.n_regions)
        elif mode in {"regression", "binary_classification"}:
            assert labels is not None
            assert inputs.shape[0] == labels.shape[0]
            self.inputs = torch.tensor(inputs, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.float32).view(-1)
        else:
            raise ValueError(
                f"Unsupported mode '{mode}'. Must be 'pretrain', 'regression', or 'binary_classification'"
            )

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

        else:
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
        assert mode in {"pretrain", "regression", "binary_classification"}
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
        elif self.mode == "binary_classification":
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
