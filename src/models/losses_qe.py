"""
Quality Estimation Loss Head for TRM.

Uses MSE loss for regression instead of cross-entropy.
The q_halt_logits from TRM are repurposed as score predictions.
"""

from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn


class QERegressionLossHead(nn.Module):
    """
    Loss head for Quality Estimation regression task.

    Changes from ACTLossHead:
    - Uses MSE loss instead of cross-entropy for score prediction
    - Outputs a single regression score per sequence
    - Metrics: Pearson correlation, Spearman correlation, MAE, RMSE
    """

    def __init__(self, model: nn.Module, loss_type: str = "mse"):
        super().__init__()
        self.model = model
        self.loss_type = loss_type

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        """
        Forward pass for QE regression.

        Expected batch keys:
        - inputs: Input token IDs [B, L]
        - scores: Target DA scores [B] (normalized 0-1)
        - lang_pair_ids: Language pair identifiers [B]
        """
        new_carry, outputs = self.model(**model_kwargs)

        # Get regression predictions (repurposed q_halt_logits)
        predicted_scores = outputs["q_halt_logits"]

        # Get targets
        batch = new_carry.current_data
        target_scores = batch["scores"].to(predicted_scores.dtype)
        lang_pair_ids = batch.get("lang_pair_ids", None)

        # Compute loss
        pred_probs = torch.sigmoid(predicted_scores)

        if self.loss_type == "mse":
            loss = F.mse_loss(pred_probs, target_scores, reduction="sum")
        elif self.loss_type == "huber":
            loss = F.huber_loss(pred_probs, target_scores, reduction="sum", delta=0.1)
        else:
            loss = F.mse_loss(pred_probs, target_scores, reduction="sum")

        # Compute metrics
        with torch.no_grad():
            batch_size = target_scores.shape[0]
            pred_probs = torch.sigmoid(predicted_scores)

            mae = torch.abs(pred_probs - target_scores).sum()
            rmse_sq = ((pred_probs - target_scores) ** 2).sum()

            pred_mean = pred_probs.mean()
            target_mean = target_scores.mean()
            pred_centered = pred_probs - pred_mean
            target_centered = target_scores - target_mean

            cov = (pred_centered * target_centered).sum()
            pred_std = torch.sqrt((pred_centered ** 2).sum() + 1e-8)
            target_std = torch.sqrt((target_centered ** 2).sum() + 1e-8)
            pearson = cov / (pred_std * target_std + 1e-8)

            metrics = {
                "count": torch.tensor(batch_size, device=target_scores.device, dtype=torch.float32),
                "qe_loss": loss.detach(),
                "mae": mae,
                "rmse_sq": rmse_sq,
                "pearson_num": cov,
                "pearson_pred_var": pred_centered.pow(2).sum(),
                "pearson_target_var": target_centered.pow(2).sum(),
                "pred_mean": pred_probs.sum(),
                "target_mean": target_scores.sum(),
                "halted_count": new_carry.halted.sum().float(),
                "steps": new_carry.steps.float().sum(),
            }

            # Per-language metrics
            if lang_pair_ids is not None:
                unique_ids = lang_pair_ids.unique()
                for lp_id in unique_ids:
                    mask = lang_pair_ids == lp_id
                    if mask.sum() > 0:
                        lp_pred = pred_probs[mask]
                        lp_target = target_scores[mask]
                        lp_mae = torch.abs(lp_pred - lp_target).sum()
                        lp_count = mask.sum().float()

                        metrics[f"lang{lp_id.item()}_mae"] = lp_mae
                        metrics[f"lang{lp_id.item()}_count"] = lp_count

        outputs["predicted_scores"] = pred_probs.detach()
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, loss, metrics, detached_outputs, new_carry.halted.all()
