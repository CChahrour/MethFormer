import datetime
import os

import torch
import wandb
from datasets import load_from_disk
from sklearn.metrics import mean_absolute_error, mean_squared_error
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

run_name = f"mf_{datetime.datetime.now().strftime('%Y-%m-%d_%H%M')}"
print(f"Run name: {run_name}")

out_dir = "/home/ubuntu/project/MethFormer/output/methformer_pretrained/"
os.makedirs(out_dir, exist_ok=True)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

dataset = load_from_disk("/home/ubuntu/project/MethFormer/data/methformer_pretrain_binned")
train_dataset = dataset["train"].shuffle(seed=42)
eval_dataset = dataset["validation"]

data_collator = MethformerCollator()

config = PretrainedConfig(
    input_dim=2,
    hidden_dim=128,
    num_hidden_layers=12,
    num_attention_heads=8,
    hidden_dropout_prob=0.1,
)

model = Methformer(config)
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
    logging_steps=1000,
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


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)
    mask = labels != -1.0
    masked_logits = logits[mask].cpu.numpy()
    masked_labels = labels[mask].cpu.numpy()
    mse = mean_squared_error(masked_labels, masked_logits)
    mae = mean_absolute_error(masked_labels, masked_logits)
    return {
        "masked_mse": mse,
        "masked_mae": mae,
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

print("Starting training...")

wandb.init(
    group="methformer_pretrain",
    job_type="pretrain_full",
    name=run_name,
    dir=out_dir,
    reinit="finish_previous",
    config=config.to_dict(),
)

trainer.train()
print("Training complete. Saving model...")

save_path = f"{out_dir}/model"
os.makedirs(save_path, exist_ok=True)
trainer.save_model(save_path)
model.config.save_pretrained(save_path)
print(f"Model saved to {save_path}")

wandb.finish()
