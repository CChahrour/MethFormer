import datetime
import json
import os

import torch
import wandb
from datasets import load_from_disk
from transformers import (
    EarlyStoppingCallback,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
)

from methformer import Methformer, MethformerCollator


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)

    # Only evaluate masked positions (label == -1.0 was masked during input)
    mask = labels != -1.0

    masked_mse = torch.mean((logits[mask] - labels[mask]) ** 2).item()
    masked_mae = torch.mean(torch.abs(logits[mask] - labels[mask])).item()

    return {
        "masked_mse": masked_mse,
        "masked_mae": masked_mae,
    }

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

dataset = load_from_disk("data/methformer_pretrain_binned")
train_dataset = dataset["train"].shuffle(seed=42)
eval_dataset = dataset["validation"]


def train():
    wandb.init(
        group="methformer_pretrain_sweep",
        job_type="pretrain_sweep",
        name=f"mf_{datetime.datetime.now().strftime('%Y-%m-%d_%H%M')}",
        dir="output/methformer_pretrain_sweep",
        reinit="finish_previous",
    )
    config = wandb.config

    run_name = f"mf_{datetime.datetime.now().strftime('%Y-%m-%d_%H%M')}"
    out_dir = f"output/methformer_pretrain_sweep/{run_name}"
    os.makedirs(out_dir, exist_ok=True)

    model_config = PretrainedConfig(
        input_dim=2,
        hidden_dim=config.hidden_dim,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        hidden_dropout_prob=config.hidden_dropout_prob,
    )

    model = Methformer(model_config)
    model.to(device)

    training_args = TrainingArguments(
        run_name=run_name,
        output_dir=os.path.join(out_dir, "checkpoints"),
        eval_on_start=True,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=256,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        learning_rate=1e-5,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        num_train_epochs=20,
        logging_dir=os.path.join(out_dir, "logs"),
        save_strategy="steps",
        save_total_limit=1,
        eval_strategy="steps",
        logging_steps=500,
        eval_steps=5000,
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=MethformerCollator(masking_ratio=config.masking_ratio),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    # Save the final model
    model.save_pretrained(os.path.join(out_dir, "model"))
    model.config.save_pretrained(os.path.join(out_dir, "model"))


with open("config/pretrain_sweep_config.json", "r") as f:
    sweep_config = json.load(f)

sweep_id = wandb.sweep(
    sweep=sweep_config,
    project="MethFormer",
)

wandb.agent(sweep_id, train, count=20)

# After the sweep
api = wandb.Api()

sweep_path = f"{wandb.run.entity}/{wandb.run.project}/{sweep_id}"
sweep = api.sweep(sweep_path)

# Filter only finished runs with masked_r2
runs = [
    run for run in sweep.runs if run.state == "finished" and "masked_r2" in run.summary
]

# Find best run by highest masked_r2
best_run = max(runs, key=lambda r: r.summary["masked_r2"])

# Save best config
best_config = {k: v for k, v in best_run.config.items() if not k.startswith("_")}
with open("best_config.json", "w") as f:
    json.dump(best_config, f, indent=2)

print(f"Best run ID: {best_run.id}")
print(f"Best masked_r2: {best_run.summary['masked_r2']}")
