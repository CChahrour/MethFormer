import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import IntegratedGradients
from datasets import load_from_disk
from methformer import Methformer, MethformerCollator
from torch.utils.data import DataLoader

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
model_root = (
    "/home/ubuntu/project/MethFormer/output/methformer_finetuned_bincls_focal_05"
)
data_dir = "/home/ubuntu/project/MethFormer/data/methformer_dataset_scaled"

model_dir = os.path.join(model_root, "model")
output_dir = os.path.join(model_root, "captum_outputs")
os.makedirs(output_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32
target_label = 1  # class index for attribution

# ─────────────────────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────────────────────
model = Methformer.from_pretrained_encoder(
    path=model_dir,
    mode="binary_classification",
    use_cls_token=False,
)
model.eval().to(device)

# ─────────────────────────────────────────────────────────────
# Load Test Dataset
# ─────────────────────────────────────────────────────────────
dataset = load_from_disk(data_dir)["test"]

def binarize_labels(example):
    example["labels"] = int(example["labels"] > 0.5)
    return example

dataset = dataset.map(binarize_labels)

dataset = dataset.map(lambda x: {"input_values": x["inputs"]})
dataset = dataset.remove_columns("inputs")

data_collator = MethformerCollator(mode="binary_classification")
dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator
)

# ─────────────────────────────────────────────────────────────
# Attribution Setup
# ─────────────────────────────────────────────────────────────
def forward_fn(input_values, attention_mask):
    out = model(input_values=input_values, attention_mask=attention_mask)
    return torch.sigmoid(out.logits).unsqueeze(1)


ig = IntegratedGradients(forward_fn)

all_attributions = []
all_inputs = []
all_preds = []
all_labels = []

# ─────────────────────────────────────────────────────────────
# Attribution Loop
# ─────────────────────────────────────────────────────────────
for batch in dataloader:
    input_values = batch["input_values"].to(device).requires_grad_()
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    # Compute attributions
    attributions, delta = ig.attribute(
        inputs=input_values,
        additional_forward_args=(attention_mask,),
        target=target_label,
        return_convergence_delta=True,
    )

    # Store results
    all_attributions.append(attributions.detach().cpu().numpy())
    all_inputs.append(input_values.detach().cpu().numpy())
    with torch.no_grad():
        preds = model(input_values=input_values, attention_mask=attention_mask).logits
        all_preds.append(torch.sigmoid(preds).cpu().numpy())
    all_labels.append(labels.cpu().numpy())



# ─────────────────────────────────────────────────────────────
# Plot examples of true positive attributions
# ─────────────────────────────────────────────────────────────

# Concatenate all batched outputs
inputs = np.concatenate(all_inputs)
attributions = np.concatenate(all_attributions)
predictions = np.concatenate(all_preds).squeeze()
labels = np.concatenate(all_labels).astype(int)


# Define threshold and mask TPs for class 1 ("Bound")
threshold = 0.5
preds = (predictions > threshold).astype(int)
mask = (labels == 1) & (preds == 1)  # True Positives

print(f"Found {mask.sum()} true positives.")

# Plot a few true positive examples
n_plot = 5
tp_inputs = inputs[mask][:n_plot]
tp_attributions = attributions[mask][:n_plot]


for i in range(n_plot):
    fig, axes = plt.subplots(tp_inputs.shape[2], 1, figsize=(12, 2.5 * tp_inputs.shape[2]), sharex=True)

    for ch in range(tp_inputs.shape[2]):
        ax = axes[ch]
        ax.plot(tp_inputs[i, :, ch], label=f"Input Channel {ch}", color='black')
        ax.plot(tp_attributions[i, :, ch], label="Attribution", color='red', alpha=0.7)
        ax.set_ylabel("Signal / Attribution")
        ax.legend(loc="upper right")
    top_bins = np.argsort(tp_attributions[i, :, ch])[-3:]
    for b in top_bins:
        ax.axvline(b, color='blue', linestyle='--', alpha=0.3)

    plt.suptitle(f"True Positive Bound Example {i+1}", fontsize=16, fontweight='bold')
    plt.xlabel("Genomic Bins")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"tp_bound_example_{i+1}.png"), dpi=300)


# ─────────────────────────────────────────────────────────────
# Save Outputs
# ─────────────────────────────────────────────────────────────
np.savez_compressed(
    os.path.join(output_dir, "captum_attributions.npz"),
    inputs=np.concatenate(all_inputs),
    attributions=np.concatenate(all_attributions),
    predictions=np.concatenate(all_preds),
    labels=np.concatenate(all_labels),
)

print(f"✅ Captum feature attributions saved to: {output_dir}")
