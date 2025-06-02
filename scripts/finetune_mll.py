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

from methformer import (
    Methformer,
    MethformerCollator,
)

# ─────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────
logger.remove()
os.makedirs("logs", exist_ok=True)
log_file = f"logs/finetune_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger.add(
    log_file, level="INFO", format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}"
)
logger.info("Starting Methformer fine-tuning...")

# ─────────────────────────────────────────────────────────────
# Run Config
# ─────────────────────────────────────────────────────────────
run_name = f"mf_ft_{datetime.datetime.now().strftime('%Y-%m-%d_%H%M')}"
output_root = "/home/ubuntu/project/MethFormer/output/methformer_finetuned/"
os.makedirs(output_root, exist_ok=True)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
logger.info(f"Using device: {device}")

# ─────────────────────────────────────────────────────────────
# Dataset Loading
# ─────────────────────────────────────────────────────────────
dataset_path = "/home/ubuntu/project/MethFormer/data/methformer_pretrain_binned"
logger.info(f"Loading dataset from {dataset_path}")
dataset = load_from_disk(dataset_path)
train_dataset = dataset["train"]
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

# Load Methformer model with regression head
model = Methformer(config, task="regression", use_cls_token=True).to(device)
logger.info("Regression model instantiated.")

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
    learning_rate=5e-5,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    num_train_epochs=10,
    logging_dir=os.path.join(output_root, "logs"),
    save_strategy="steps",
    save_total_limit=2,
    evaluation_strategy="steps",
    logging_steps=200,
    eval_steps=500,
    save_steps=1000,
    metric_for_best_model="mse",
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

    mse = torch.mean((logits - labels) ** 2).item()
    mae = torch.mean(torch.abs(logits - labels)).item()
    r2 = 1 - torch.sum((logits - labels) ** 2) / torch.sum(
        (labels - labels.mean()) ** 2
    )

    return {
        "mse": mse,
        "mae": mae,
        "r2": r2.item(),
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
    group="finetune_methformer",
    job_type="finetune_mll",
    name=run_name,
    dir=output_root,
    reinit="finish_previous",
    config=config.to_dict(),
)

# ─────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────
logger.info("Starting fine-tuning...")
trainer.train()
logger.info("Fine-tuning complete.")

# ─────────────────────────────────────────────────────────────
# Save Final Model
# ─────────────────────────────────────────────────────────────
save_path = os.path.join(output_root, "model")
os.makedirs(save_path, exist_ok=True)
trainer.save_model(save_path)
model.config.save_pretrained(save_path)
logger.info(f"✅ Fine-tuned model saved to {save_path}")

wandb.finish()
