# 🧚 MethFormer: A Transformer for Binned DNA Methylation

**MethFormer** is a masked regression transformer model trained to learn local and long-range patterns in DNA methylation (5mC and 5hmC) across genomic regions. Pretrained on binned methylation data, it is designed for downstream fine-tuning on tasks such as predicting MLL binding or chromatin state.

---

## 🚀 Overview

* **Inputs**: Binned methylation values (5mC, 5hmC) over 1024bp windows (32 bins × 2 channels)
* **Pretraining objective**: Masked methylation imputation (per-bin regression)
* **Architecture**: Transformer encoder with linear projection head
* **Downstream tasks**: MLL binding prediction, chromatin state inference, or enhancer classification

---

## 📁 Project Structure

```
...
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

## 🔍 Metrics

* `masked_mse`: Mean squared error over unmasked positions
* `masked_mae`: Mean absolute error

---

## 🧪 Fine-tuning on MLL Binding

After pretraining:

1. Replace the regression head with a scalar head for MLL prediction.
2. Use a `Trainer` to fine-tune on log1p-transformed MLL-N RPKM values mean over 1kb regions.

See `scripts/finetune_mll.py` for an example.

---

## 📊 Visualizations & Interpretability

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
