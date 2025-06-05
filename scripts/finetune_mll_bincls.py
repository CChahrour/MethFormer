import datetime
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from datasets import load_from_disk
from loguru import logger
from methformer import (
    Methformer,
    MethformerCollator,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

import wandb

# ─────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────
logger.remove()
os.makedirs("logs", exist_ok=True)
log_file = "logs/finetune_mll.log"
logger.add(
    log_file, level="INFO", format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}"
)
logger.info("Starting Methformer fine-tuning...")

# ─────────────────────────────────────────────────────────────
# Run Config
# ─────────────────────────────────────────────────────────────
run_name = f"mf_ft_{datetime.datetime.now().strftime('%Y-%m-%d_%H%M')}"
output_root = "/home/ubuntu/project/MethFormer/output/methformer_finetuned_bincls/"
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
dataset_path = "/home/ubuntu/project/MethFormer/data/methformer_dataset_scaled"
logger.info(f"Loading dataset from {dataset_path}")
dataset = load_from_disk(dataset_path)
train_dataset = dataset["train"].shuffle(seed=42)
eval_dataset = dataset["validation"]
test_dataset = dataset["test"]

data_collator = MethformerCollator()

# ─────────────────────────────────────────────────────────────
# Model Configuration
# ─────────────────────────────────────────────────────────────
# Load Methformer model with binary_classification head
model_pretrained_path = "/home/ubuntu/project/MethFormer/output/methformer_pretrained/model"
model = Methformer.from_pretrained_encoder(
    path=model_pretrained_path,
    mode="binary_classification",
    use_cls_token=False
)
logger.info("binary_classification model instantiated.")

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
    learning_rate=1e-6,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    num_train_epochs=20,
    logging_dir=os.path.join(output_root, "logs"),
    save_strategy="steps",
    save_total_limit=2,
    eval_strategy="steps",
    logging_steps=200,
    eval_steps=500,
    save_steps=1000,
    metric_for_best_model="f1",
    greater_is_better=True,
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

    # Ensure proper shape
    if logits.ndim == 2 and logits.shape[1] == 1:
        probs = torch.sigmoid(logits.squeeze(1))
    else:
        probs = torch.sigmoid(logits)

    preds = (probs > 0.5).long()

    # Ensure both are int
    labels = labels.long()

    accuracy = accuracy_score(labels.cpu(), preds.cpu())
    precision = precision_score(labels.cpu(), preds.cpu(), zero_division=0)
    recall = recall_score(labels.cpu(), preds.cpu(), zero_division=0)
    f1 = f1_score(labels.cpu(), preds.cpu(), zero_division=0)

    try:
        roc_auc = roc_auc_score(labels.cpu(), probs.cpu())
    except ValueError:
        roc_auc = float("nan")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
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
    job_type="finetune_mll_bincls",
    name=run_name,
    dir=output_root,
    reinit="finish_previous",
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

# ─────────────────────────────────────────────────────────────
# Predict on Test Set
# ─────────────────────────────────────────────────────────────
logger.info("Evaluating on test set...")
predictions = trainer.predict(test_dataset=test_dataset)
# pickle the predictions

with open(os.path.join(output_root, "test_predictions.pkl"), "wb") as f:
    pickle.dump(predictions, f)

preds = predictions.predictions.flatten()
labels = predictions.label_ids.flatten()

scaler_path = "/home/ubuntu/project/MethFormer/data/mll_scaler.pkl"


logger.info("Test set evaluation complete.")
wandb.finish()
