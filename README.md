# 🧚 MethFormer: A Transformer for DNA Methylation


🤗 [CChahrour/Methformer](https://huggingface.co/CChahrour/Methformer)

**MethFormer**  is a Vision Transformer (ViT)-inspired model for DNA methylation modeling. It uses a masked regression objective to learn both local and long-range patterns in 5mC and 5hmC methylation across genomic regions. Pretrained on binned methylation data, MethFormer is designed for downstream fine-tuning on tasks such as predicting MLL binding or chromatin state.

---

## 🚀 Overview

* **Inputs**: Binned methylation values (5mC, 5hmC) over 1024bp windows (32 bins × 2 channels)
* **Pretraining objective**: Masked methylation imputation (per-bin regression)
* **Architecture**: Transformer encoder with linear projection head
* **Downstream tasks**: MLL binding prediction, chromatin state inference, or enhancer classification

---

## 📁 Project Structure

```
.
├── config/                       # config
├── data/                         # Binned methylation datasets (HuggingFace format)
├── output/                       # Pretrained models, logs, and checkpoints
├── scripts/                      
│   ├── methformer.py             # Model classes, data collator, 
│   ├── pretrain_methformer.py    # Main training script
│   └── finetune_mll.py           # (optional) downstream fine-tuning
├── requirements.txt
└── README.md
```

---

## 👩‍💻 Pretraining MethFormer

### Step 1: Prepare Dataset

Preprocess 5mC and 5hmC data into 1024bp windows, binned into 32 bins × 2 features. Save using Hugging Face's `datasets.DatasetDict` format:

```
DatasetDict({
  train: Dataset({
    features: ['input_values', 'attention_mask', 'labels']
  }),
  validation: Dataset(...)
})
```

### Step 2: Run Pretraining

```bash
python scripts/pretrain_methformer.py
```

Options can be customized inside the script or modified for sweep tuning. This will:

* Train the model using masked regression loss
* Evaluate on a held-out chromosome (e.g., `chr8`)
* Log metrics to [Weights & Biases](https://wandb.ai)
* Save the best model checkpoint

---

## 📊 Metrics

* `masked_mse`: Mean squared error over unmasked positions
* `masked_mae`: Mean absolute error

---

## 🧪 Fine-tuning on MLL Binding

After pretraining:

1. Replace the regression head with a scalar head for MLL prediction.
2. Use a `Trainer` to fine-tune on log1p-transformed MLL-N RPKM values mean over 1kb regions.

```
input_values: (N, 32, 2)    # Methylation input per region, 32 bins over a 1kb region (32bp each), 2 channels: 5mC and 5hmC, Values range from 0–1 (methylation fraction), or -1 if missing.
attention_mask: (N, 32)     # Binary mask that indicates which bins have valid methylation values
labels: (N,)                # Scalar target per region, MLL-N mean RPKM over 1kb regions
```

See `scripts/finetune_mll.py` for an example.

---

## 🔍 Visualizations & Interpretability

You can run [Captum](https://captum.ai) or SHAP for:

* Per-bin attribution of 5mC/5hmC to MLL binding
* Visualizing what MethFormer attends to during fine-tuning

---

## 🛠️ Dependencies

Key packages:

* `transformers`
* `datasets`
* `wandb`
* `torch`
* `anndata`
* `scikit-learn`

---

## 🧠 Acknowledgements

* Built with inspiration from DNABERT, Grover, and vision transformers
