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
from scipy.stats import pearsonr, spearmanr
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
output_root = "/home/ubuntu/project/MethFormer/output/methformer_finetuned_scaled/"
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
# Load Methformer model with regression head
model_pretrained_path = "/home/ubuntu/project/MethFormer/output/methformer_pretrained/model"
model = Methformer.from_pretrained_encoder(
    path=model_pretrained_path,
    mode="regression",
    use_cls_token=False
)
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
    # Do NOT mask here — all labels are known
    mse = torch.mean((logits - labels) ** 2).item()
    mae = torch.mean(torch.abs(logits - labels)).item()
    rmse = torch.sqrt(torch.mean((logits - labels) ** 2)).item()
    # R²
    r2 = 1 - torch.sum((logits - labels) ** 2) / torch.sum(
        (labels - labels.mean()) ** 2
    )
    r2 = r2.item()
    # Correlations
    logits_np = logits.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    pearson_corr, _ = pearsonr(logits_np, labels_np)
    spearman_corr, _ = spearmanr(logits_np, labels_np)

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson": pearson_corr,
        "spearman": spearman_corr,
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
# Load the scaler
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)
# Inverse transform the predictions and labels
preds = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
labels = scaler.inverse_transform(labels.reshape(-1, 1)).flatten()

# plot predictions vs labels
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=preds,
    y=labels,
    alpha=0.4,
    edgecolor=None,
    s=10,
)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(
    [0, 1], [0, 1], color="red", linestyle="--", linewidth=1, label="Perfect Prediction"
)

# Labels and title
plt.xlabel("Predicted MLL binding", fontsize=12, fontweight="bold")
plt.ylabel("True MLL binding", fontsize=12, fontweight="bold")
plt.title(
    "Predictions vs Labels",
    fontsize=14,
    fontweight="bold",
)
plt.legend()
plt.tight_layout()

# Save
plot_path = os.path.join(output_root, "predictions_vs_labels.png")
plt.savefig(plot_path, dpi=600)
plt.close()

# log test set metrics to wandb
logger.info("Logging test set metrics to Weights & Biases...")
wandb.run.summary["predictions_vs_labels_plot"] = wandb.Image(
    os.path.join(output_root, "predictions_vs_labels.png")
)

logger.info("Test set evaluation complete.")
wandb.finish()
