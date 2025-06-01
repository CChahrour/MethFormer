import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput


class MethformerDataset(Dataset):
    """
    Dataset that returns masked inputs, original labels, and attention masks.
    """

    def __init__(
        self, data_tensor, chunk_size=128, mask_value=-1.0, masking_ratio=0.15
    ):
        self.data = data_tensor
        self.n_samples, self.n_regions, self.n_channels = self.data.shape
        self.chunk_size = min(chunk_size, self.n_regions)
        self.mask_value = mask_value
        self.masking_ratio = masking_ratio

    def __len__(self):
        return self.n_samples * (self.n_regions // self.chunk_size)

    def __getitem__(self, idx):
        sample_idx = idx % self.n_samples
        chunk_start = random.randint(0, self.n_regions - self.chunk_size)
        chunk = self.data[sample_idx, chunk_start : chunk_start + self.chunk_size, :]

        x = torch.tensor(chunk, dtype=torch.float32)
        mask = torch.rand(self.chunk_size) < self.masking_ratio
        x_masked = x.clone()
        x_masked[mask] = self.mask_value

        return {"inputs": x_masked, "labels": x, "attention_mask": ~mask}


class MethformerCollator:
    def __init__(self, masking_ratio=0.15):
        self.masking_ratio = masking_ratio

    def __call__(self, batch):
        def ensure_tensor(x):
            if isinstance(x, torch.Tensor):
                return x
            return torch.tensor(x, dtype=torch.float32)

        inputs = [ensure_tensor(item["inputs"]) for item in batch]
        labels = [ensure_tensor(item["labels"]) for item in batch]
        attention_mask = [
            torch.tensor(item["attention_mask"], dtype=torch.bool) for item in batch
        ]

        inputs_tensor = torch.stack(inputs)
        labels_tensor = torch.stack(labels)
        attention_mask_tensor = torch.stack(attention_mask)

        return {
            "input_values": inputs_tensor,
            "labels": labels_tensor,
            "attention_mask": attention_mask_tensor,
        }


class Methformer(PreTrainedModel):
    """
    Masked Transformer model for methylation data.
    """

    def __init__(self, config):
        super().__init__(config)
        self.input_dim = getattr(config, "input_dim", 2)
        hidden_dim = getattr(config, "hidden_dim", 128)
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        dropout = config.hidden_dropout_prob
        max_len = getattr(config, "max_position_embeddings", 1024)

        self.embed = nn.Linear(self.input_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(hidden_dim, self.input_dim)

    def forward(self, input_values, attention_mask, labels=None):
        x = self.embed(input_values)
        x = x + self.pos_embed[:, : x.size(1), :].to(x.device)

        attn_mask = ~attention_mask.bool()
        x = self.encoder(x, src_key_padding_mask=attn_mask)
        output = self.output_head(x)

        loss = None
        if labels is not None:
            mask = attention_mask.unsqueeze(-1).expand_as(labels)
            loss_fn = nn.MSELoss()
            loss = loss_fn(output[mask], labels[mask])

        return ModelOutput(loss=loss, last_hidden_state=output)


class MethformerRegressor(PreTrainedModel):
    """
    Regression model that uses Methformer as the encoder.
    """

    def __init__(self, config):
        super().__init__(config)
        self.encoder = Methformer(config)
        self.regression_head = nn.Linear(config.hidden_dim, 1)

    def forward(self, input_values, attention_mask, labels=None):
        x = self.encoder(input_values, attention_mask)
        pooled = (x * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            1, keepdim=True
        )
        logits = self.regression_head(pooled)
        loss = None
        if labels is not None:
            loss = F.mse_loss(logits, labels)
        return {"loss": loss, "logits": logits}
