#!/usr/bin/env python3
"""
Standalone evaluation script for QE checkpoints.
Computes per-language Pearson and Spearman correlations with bootstrap confidence intervals.

Usage:
    # Basic evaluation
    python src/eval.py --checkpoint checkpoints/TRM-QE/exp/latest --config-name frozen_xlmr_L4

    # With bootstrap confidence intervals (recommended for paper)
    python src/eval.py --checkpoint checkpoints/TRM-QE/exp/latest --config-name frozen_xlmr_L4 --bootstrap 1000

    # Evaluate on CPU
    python src/eval.py --checkpoint ... --config-name frozen_xlmr_L4 --device cpu

    # Save detailed predictions for error analysis
    python src/eval.py --checkpoint ... --save-predictions predictions.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
from pathlib import Path

import torch
import numpy as np
from scipy import stats
import hydra
from omegaconf import OmegaConf

from data.qe_dataset import QEPuzzleDataset, QEPuzzleDatasetConfig, QEDatasetMetadata
from models.trm import TinyRecursiveReasoningModel_ACTV1
from models.losses_qe import QERegressionLossHead


# Language pair names for display
LANG_PAIR_NAMES = {
    0: "en-gu (Gujarati)",
    1: "en-hi (Hindi)",
    2: "en-mr (Marathi)",
    3: "en-ta (Tamil)",
    4: "en-te (Telugu)",
    5: "et-en (Estonian)",
    6: "ne-en (Nepali)",
    7: "si-en (Sinhala)",
}


def get_device(device_str: str = "auto") -> torch.device:
    """Auto-detect best available device or use specified one."""
    if device_str and device_str != "auto":
        return torch.device(device_str)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> dict:
    """
    Compute bootstrap confidence intervals for Pearson and Spearman correlations.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    pearson_samples = []
    spearman_samples = []

    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        y_t = y_true[indices]
        y_p = y_pred[indices]

        try:
            r, _ = stats.pearsonr(y_t, y_p)
            if not np.isnan(r):
                pearson_samples.append(r)
        except:
            pass

        try:
            rho, _ = stats.spearmanr(y_t, y_p)
            if not np.isnan(rho):
                spearman_samples.append(rho)
        except:
            pass

    alpha = (1 - ci) / 2

    result = {}

    if pearson_samples:
        result['pearson'] = {
            'mean': float(np.mean(pearson_samples)),
            'ci_lower': float(np.percentile(pearson_samples, alpha * 100)),
            'ci_upper': float(np.percentile(pearson_samples, (1 - alpha) * 100)),
        }

    if spearman_samples:
        result['spearman'] = {
            'mean': float(np.mean(spearman_samples)),
            'ci_lower': float(np.percentile(spearman_samples, alpha * 100)),
            'ci_upper': float(np.percentile(spearman_samples, (1 - alpha) * 100)),
        }

    return result


def load_model_from_checkpoint(checkpoint_path: str, config: dict, metadata: QEDatasetMetadata, device: torch.device):
    """Load model from checkpoint."""
    model_cfg = dict(
        **{k: v for k, v in config['arch'].items() if k not in ['name', 'loss']},
        batch_size=config.get('global_batch_size', 256),
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
        causal=False
    )

    # Set device for layers.py PretrainedEncoder
    os.environ["TRM_DEVICE"] = str(device)

    with torch.device(device):
        model = TinyRecursiveReasoningModel_ACTV1(model_cfg)
        loss_config = {k: v for k, v in config['arch']['loss'].items() if k != 'name'}
        model = QERegressionLossHead(model, **loss_config)

    model = model.to(device)

    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, assign=True)
    model.eval()

    return model


def evaluate_checkpoint(
    model,
    test_loader,
    metadata,
    device: torch.device,
    n_bootstrap: int = 0,
    return_predictions: bool = False
):
    """Run evaluation and compute all metrics."""
    all_preds = []
    all_targets = []
    all_lang_ids = []

    with torch.inference_mode():
        for set_name, batch, global_batch_size in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.device(device):
                carry = model.initial_carry(batch)

            while True:
                carry, loss, metrics, preds, all_finish = model(
                    carry=carry, batch=batch, return_keys=["predicted_scores"]
                )
                if all_finish:
                    break

            if "predicted_scores" in preds:
                all_preds.append(preds["predicted_scores"].cpu())
            all_targets.append(batch["scores"].cpu())
            if "lang_pair_ids" in batch:
                all_lang_ids.append(batch["lang_pair_ids"].cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_lang_ids = torch.cat(all_lang_ids).numpy() if all_lang_ids else None

    # Overall metrics
    pearson, _ = stats.pearsonr(all_preds, all_targets)
    spearman, _ = stats.spearmanr(all_preds, all_targets)
    mae = np.abs(all_preds - all_targets).mean()
    rmse = np.sqrt(((all_preds - all_targets) ** 2).mean())

    results = {
        "overall": {
            "pearson": float(pearson),
            "spearman": float(spearman),
            "mae": float(mae),
            "rmse": float(rmse),
            "n_samples": len(all_preds),
        },
        "per_language": {}
    }

    # Bootstrap CI for overall metrics
    if n_bootstrap > 0:
        print(f"Computing bootstrap CIs with {n_bootstrap} samples...")
        ci_results = bootstrap_ci(all_targets, all_preds, n_bootstrap=n_bootstrap)
        if 'pearson' in ci_results:
            results["overall"]["pearson_ci"] = [
                ci_results['pearson']['ci_lower'],
                ci_results['pearson']['ci_upper']
            ]
        if 'spearman' in ci_results:
            results["overall"]["spearman_ci"] = [
                ci_results['spearman']['ci_lower'],
                ci_results['spearman']['ci_upper']
            ]

    # Per-language metrics
    if all_lang_ids is not None:
        for lang_id in np.unique(all_lang_ids):
            mask = all_lang_ids == lang_id
            n_samples = mask.sum()

            if n_samples > 10:
                lp_preds = all_preds[mask]
                lp_targets = all_targets[mask]

                lp_pearson, _ = stats.pearsonr(lp_preds, lp_targets)
                lp_spearman, _ = stats.spearmanr(lp_preds, lp_targets)
                lp_mae = np.abs(lp_preds - lp_targets).mean()

                lang_name = LANG_PAIR_NAMES.get(int(lang_id), f"lang{int(lang_id)}")
                lang_result = {
                    "pearson": float(lp_pearson),
                    "spearman": float(lp_spearman),
                    "mae": float(lp_mae),
                    "n_samples": int(n_samples),
                }

                if n_bootstrap > 0:
                    lp_ci = bootstrap_ci(lp_targets, lp_preds, n_bootstrap=n_bootstrap)
                    if 'pearson' in lp_ci:
                        lang_result["pearson_ci"] = [
                            lp_ci['pearson']['ci_lower'],
                            lp_ci['pearson']['ci_upper']
                        ]
                    if 'spearman' in lp_ci:
                        lang_result["spearman_ci"] = [
                            lp_ci['spearman']['ci_lower'],
                            lp_ci['spearman']['ci_upper']
                        ]

                results["per_language"][lang_name] = lang_result

    if return_predictions:
        return results, all_preds, all_targets, all_lang_ids
    return results


def format_with_ci(value: float, ci: list = None) -> str:
    """Format a value with optional CI."""
    if ci:
        return f"{value:.4f} [{ci[0]:.3f}, {ci[1]:.3f}]"
    return f"{value:.4f}"


def print_results(results: dict, experiment_name: str = ""):
    """Print results in a nice table format."""
    print("\n" + "=" * 80)
    if experiment_name:
        print(f"Results for: {experiment_name}")
        print("=" * 80)

    o = results["overall"]
    has_ci = "pearson_ci" in o

    print(f"\nOverall (n={o['n_samples']}):")
    print(f"  Pearson:  {format_with_ci(o['pearson'], o.get('pearson_ci'))}")
    print(f"  Spearman: {format_with_ci(o['spearman'], o.get('spearman_ci'))}")
    print(f"  MAE:      {o['mae']:.4f}")
    print(f"  RMSE:     {o['rmse']:.4f}")

    print(f"\nPer-Language Results:")
    if has_ci:
        print("-" * 100)
        print(f"{'Language':<20} {'Pearson':>28} {'Spearman':>28} {'MAE':>10} {'N':>8}")
        print("-" * 100)
        for lang_name, m in sorted(results["per_language"].items()):
            p_str = format_with_ci(m['pearson'], m.get('pearson_ci'))
            s_str = format_with_ci(m['spearman'], m.get('spearman_ci'))
            print(f"{lang_name:<20} {p_str:>28} {s_str:>28} {m['mae']:>10.4f} {m['n_samples']:>8}")
    else:
        print("-" * 70)
        print(f"{'Language':<20} {'Pearson':>10} {'Spearman':>10} {'MAE':>10} {'N':>8}")
        print("-" * 70)
        for lang_name, m in sorted(results["per_language"].items()):
            print(f"{lang_name:<20} {m['pearson']:>10.4f} {m['spearman']:>10.4f} {m['mae']:>10.4f} {m['n_samples']:>8}")

    print("-" * (100 if has_ci else 70))
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate QE checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--config-name", type=str, default="base", help="Hydra config name (without .yaml)")
    parser.add_argument("--data-path", type=str, default=None, help="Override data path")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for results")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for evaluation")
    parser.add_argument("--bootstrap", type=int, default=0, help="Number of bootstrap samples for CI")
    parser.add_argument("--save-predictions", type=str, default=None, help="Save predictions to CSV")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, mps")

    args = parser.parse_args()

    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Set device for layers.py PretrainedEncoder
    os.environ["TRM_DEVICE"] = str(device)

    # Load config
    with hydra.initialize(config_path="../config", version_base=None):
        cfg = hydra.compose(config_name=args.config_name)
        config = OmegaConf.to_container(cfg, resolve=True)

    if args.data_path:
        config['data_paths'] = [args.data_path]

    # Create test dataloader
    data_config = QEPuzzleDatasetConfig(
        seed=config.get('seed', 0),
        dataset_paths=config['data_paths'],
        rank=0,
        num_replicas=1,
        epochs_per_iter=1,
        global_batch_size=args.batch_size,
        test_set_mode=True,
    )
    test_dataset = QEPuzzleDataset(data_config, split="test")
    metadata = test_dataset.metadata

    # Load model
    model = load_model_from_checkpoint(args.checkpoint, config, metadata, device)

    # Evaluate
    if args.save_predictions:
        results, all_preds, all_targets, all_lang_ids = evaluate_checkpoint(
            model, test_dataset, metadata, device, n_bootstrap=args.bootstrap, return_predictions=True
        )
    else:
        results = evaluate_checkpoint(model, test_dataset, metadata, device, n_bootstrap=args.bootstrap)

    exp_name = Path(args.checkpoint).parent.name
    print_results(results, exp_name)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")

    if args.save_predictions:
        import pandas as pd
        errors = all_preds - all_targets
        lang_names = [LANG_PAIR_NAMES.get(int(lid), f"lang{int(lid)}") for lid in all_lang_ids]
        df = pd.DataFrame({
            'lang_pair': lang_names,
            'true_score': all_targets,
            'pred_score': all_preds,
            'error': errors,
            'abs_error': np.abs(errors)
        })
        df.to_csv(args.save_predictions, index=False)
        print(f"Predictions saved to: {args.save_predictions}")

    return results


if __name__ == "__main__":
    main()
