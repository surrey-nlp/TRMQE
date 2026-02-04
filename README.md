# TRM-QE: Parameter-Efficient Quality Estimation with Frozen Embeddings

Code for **"Frozen Embeddings for Parameter-Efficient Quality Estimation"**

## Key Findings

| Model | Trainable Params | Spearman |
|-------|------------------|----------|
| TRM (frozen XLM-R, L=4) | **7M** | **0.370** |
| TRM (fine-tuned XLM-R, L=2) | 262M | 0.369 |
| Standard 8-layer (frozen) | 27M | 0.290 |
| MonoTransQuest | 560M | ~0.35 |

**Main findings:**
- Frozen XLM-R embeddings **match** fine-tuned performance (0.370 vs 0.369 Spearman)
- **37x fewer trainable parameters** (7M vs 262M)
- Weight sharing is **critical**: frozen TRM (0.370) vs frozen standard 8-layer (0.290)
- Outperforms MonoTransQuest on Hindi (+0.097) and Tamil (+0.039) with **80x fewer params**

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Build the dataset from [Surrey Low-Resource QE Dataset](https://huggingface.co/datasets/surrey-nlp/Low-resource-QE-DA-dataset):

```bash
# For XLM-R experiments (main results)
python src/data/build_qe_dataset.py \
    --output-dir data/qe-lowres-xlmr \
    --tokenizer xlm-roberta-large

# For mT5 experiments (optional)
python src/data/build_qe_dataset.py \
    --output-dir data/qe-lowres \
    --tokenizer google/mt5-small
```

### 3. Train Main Model

```bash
# Main result: Frozen XLM-R with L=4 cycles
python src/train.py --config-name frozen_xlmr_L4
```

### 4. Evaluate

```bash
# With bootstrap confidence intervals
python src/eval.py \
    --checkpoint checkpoints/TRM-QE/frozen_xlmr_L4/latest \
    --config-name frozen_xlmr_L4 \
    --bootstrap 1000
```

## Experiments

### Main Results (Table 3)

| Config | Description | Paper Result |
|--------|-------------|--------------|
| `frozen_xlmr_L4` | Main: Frozen XLM-R, L=4 | 0.370 Spearman |
| `finetuned_xlmr_L2` | Baseline: Fine-tuned XLM-R, L=2 | 0.369 Spearman |

### Ablations

| Config | Description | Paper Result |
|--------|-------------|--------------|
| `frozen_xlmr_L1` | L-cycle ablation: L=1 | 0.321 Spearman |
| `frozen_xlmr_L6` | L-cycle ablation: L=6 | 0.288 Spearman |
| `frozen_xlmr_L4_steps4` | External iteration: 4 steps | Degrades |
| `frozen_mt5small_L4` | mT5-small embeddings | Lower than XLM-R |

### Baselines

| Config | Description | Paper Result |
|--------|-------------|--------------|
| `standard_frozen` | 8-layer, no weight sharing, frozen | 0.290 Spearman |
| `standard_finetuned` | 8-layer, no weight sharing, fine-tuned | 0.336 Spearman |

### Run All Experiments

```bash
# Run each config sequentially
for config in frozen_xlmr_L4 frozen_xlmr_L1 frozen_xlmr_L6 frozen_xlmr_L4_steps4 \
              frozen_mt5small_L4 finetuned_xlmr_L2 standard_frozen standard_finetuned; do
    python src/train.py --config-name $config
done
```

## Repository Structure

```
trm-qe-minimal/
├── src/
│   ├── models/
│   │   ├── trm.py          # Core TRM architecture
│   │   ├── losses_qe.py    # QE regression loss head
│   │   └── layers.py       # Embeddings, attention, RoPE, SwiGLU
│   ├── data/
│   │   ├── qe_dataset.py   # Dataset loader
│   │   └── build_qe_dataset.py  # Dataset preparation
│   ├── train.py            # Training script
│   └── eval.py             # Evaluation with bootstrap CI
├── config/
│   ├── base.yaml           # Base hyperparameters
│   ├── frozen_xlmr_L4.yaml # Main result
│   └── ...                 # Other experiment configs
└── requirements.txt
```

## Distributed Training

```bash
torchrun --nproc_per_node=4 src/train.py --config-name frozen_xlmr_L4
```

## Citation

```bibtex
@article{trm-qe-2026,
  title={Frozen Embeddings for Parameter-Efficient Quality Estimation},
  author={},
  journal={},
  year={2026}
}
```

## Acknowledgments

- Surrey NLP for the [Low-Resource QE Dataset](https://huggingface.co/datasets/surrey-nlp/Low-resource-QE-DA-dataset)
- HuggingFace for pretrained models (XLM-R, mT5)
