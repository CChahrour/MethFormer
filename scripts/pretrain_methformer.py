import os
import datetime

import torch
import wandb
from loguru import logger
from datasets import load_from_disk
from transformers import (
    EarlyStoppingCallback,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
)

from methformer import Methformer, MethformerCollator

# ─────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────
logger.remove()
os.makedirs("logs", exist_ok=True)
log_file = f"logs/pretrain_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger.add(log_file, level="INFO", format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}")
logger.info("Starting Methformer pretraining...")

# ─────────────────────────────────────────────────────────────
# Run Config
# ─────────────────────────────────────────────────────────────
run_name = f"mf_{datetime.datetime.now().strftime('%Y-%m-%d_%H%M')}"
output_root = "/home/ubuntu/project/MethFormer/output/methformer_pretrained/"
os.makedirs(output_root, exist_ok=True)

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
logger.info(f"Using device: {device}")

# ─────────────────────────────────────────────────────────────
# Dataset Loading
# ─────────────────────────────────────────────────────────────
dataset_path = "/home/ubuntu/project/MethFormer/data/methformer_pretrain_dataset"
logger.info(f"Loading dataset from {dataset_path}")
dataset = load_from_disk(dataset_path)
train_dataset = dataset["train"].shuffle(seed=42)
eval_dataset = dataset["validation"]

data_collator = MethformerCollator()

# ─────────────────────────────────────────────────────────────
# Model Configuration
# ─────────────────────────────────────────────────────────────
config = PretrainedConfig(
    input_dim=2,
    hidden_dim=128,
    num_hidden_layers=12,
    num_attention_heads=8,
    hidden_dropout_prob=0.1,
    max_position_embeddings=1024,
)

model = Methformer(config, mode="pretrain", use_cls_token=False).to(device)
logger.info("Model instantiated.")

# ─────────────────────────────────────────────────────────────
# Training Arguments
# ─────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    run_name=run_name,
    output_dir=os.path.join(output_root, "checkpoints"),
    eval_on_start=True,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=256,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
    learning_rate=1e-5,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    num_train_epochs=20,
    logging_dir=os.path.join(output_root, "logs"),
    save_strategy="steps",
    save_total_limit=1,
    eval_strategy="steps",
    logging_steps=500,
    eval_steps=1000,
    save_steps=5000,
    metric_for_best_model="masked_mse",
    greater_is_better=False,
    report_to="wandb",
    disable_tqdm=False,
    dataloader_num_workers=8,
    remove_unused_columns=False,
    fp16=not torch.backends.mps.is_available(),
    load_best_model_at_end=True,
    seed=42,
)

# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)

    mask = labels != -1.0
    masked_mse = torch.mean((logits[mask] - labels[mask]) ** 2).item()
    masked_mae = torch.mean(torch.abs(logits[mask] - labels[mask])).item()

    return {
        "masked_mse": masked_mse,
        "masked_mae": masked_mae,
    }

# ─────────────────────────────────────────────────────────────
# Trainer Setup
# ─────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# ─────────────────────────────────────────────────────────────
# Weights & Biases
# ─────────────────────────────────────────────────────────────
logger.info("Initializing Weights & Biases run...")
wandb.init(
    project="MethFormer",
    group="pretrain_methformer",
    job_type="pretrain_full",
    name=run_name,
    dir=output_root,
    reinit="finish_previous",
    config=config.to_dict(),
)

# ─────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────
logger.info("Starting training...")
trainer.train()
logger.info("Training complete.")

# ─────────────────────────────────────────────────────────────
# Save Final Model
# ─────────────────────────────────────────────────────────────
save_path = os.path.join(output_root, "model")
os.makedirs(save_path, exist_ok=True)
trainer.save_model(save_path)
model.config.save_pretrained(save_path)
logger.info(f"✅ Model saved to {save_path}")

wandb.finish()
