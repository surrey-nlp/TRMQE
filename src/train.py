#!/usr/bin/env python3
"""
Quality Estimation training script for TRM-QE.

Usage:
    # Train with frozen XLM-R embeddings (main result)
    python src/train.py --config-name frozen_xlmr_L4

    # Train on CPU (for testing)
    python src/train.py --config-name frozen_xlmr_L4 device=cpu

    # Train with fine-tuned embeddings
    python src/train.py --config-name finetuned_xlmr_L2

    # Distributed training (GPU only)
    torchrun --nproc_per_node=4 src/train.py --config-name frozen_xlmr_L4
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import math
import copy

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
import numpy as np
from scipy import stats

from data.qe_dataset import QEPuzzleDataset, QEPuzzleDatasetConfig, QEDatasetMetadata
from models.trm import TinyRecursiveReasoningModel_ACTV1
from models.losses_qe import QERegressionLossHead
from models.ema import EMAHelper

# Try importing AdamATan2
try:
    from adam_atan2 import AdamATan2
    USE_ADAM_ATAN2 = True
    print("✅ Using AdamATan2 optimizer")
except ImportError:
    print("⚠️  AdamATan2 not available, falling back to standard Adam")
    from torch.optim import Adam as AdamATan2
    USE_ADAM_ATAN2 = False

# Global device - set in launch()
DEVICE = None


def get_device(config_device: Optional[str] = None) -> torch.device:
    """Auto-detect best available device or use specified one."""
    if config_device and config_device != "auto":
        return torch.device(config_device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class QEPretrainConfig(pydantic.BaseModel):
    """Training configuration."""
    arch: ArchConfig

    # Data
    data_paths: List[str]
    data_paths_test: List[str] = []

    # Training
    global_batch_size: int
    epochs: int
    gradient_accumulation_steps: int = 1

    lr: float
    lr_min_ratio: float = 0.1
    lr_warmup_steps: int = 100

    weight_decay: float
    beta1: float = 0.9
    beta2: float = 0.99

    # Puzzle embedding
    puzzle_emb_lr: float = 1e-4
    puzzle_emb_weight_decay: float = 0.1

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    checkpoint_keep_latest_only: bool = True
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0

    ema: bool = False
    ema_rate: float = 0.999

    # Wandb
    wandb_tags: List[str] = []

    # Device selection (auto, cpu, cuda, mps)
    device: str = "auto"


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int
    accum_step: int = 0


def create_qe_dataloader(config: QEPretrainConfig, split: str, rank: int, world_size: int, epochs_per_iter: int = 1):
    """Create QE dataloader."""
    inner_config = QEPuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths_test if split == "test" and config.data_paths_test else config.data_paths,
        rank=rank,
        num_replicas=world_size,
        epochs_per_iter=epochs_per_iter,
        global_batch_size=config.global_batch_size,
        test_set_mode=(split == "test"),
    )
    dataset = QEPuzzleDataset(inner_config, split=split)

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=0 if DEVICE.type == "cpu" else 1,
        prefetch_factor=None if DEVICE.type == "cpu" else 4,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=False if DEVICE.type == "cpu" else True
    )
    return dataloader, dataset.metadata


def create_model(config: QEPretrainConfig, train_metadata: QEDatasetMetadata, rank: int, world_size: int):
    """Create TRM-QE model."""
    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False
    )

    # Create model on the target device
    with torch.device(DEVICE):
        model = TinyRecursiveReasoningModel_ACTV1(model_cfg)
        print(model)
        model = QERegressionLossHead(model, **config.arch.loss.__pydantic_extra__)

    # Move to device explicitly (needed for some device types)
    model = model.to(DEVICE)

    if rank == 0 and config.load_checkpoint:
        state_dict = torch.load(config.load_checkpoint, map_location=DEVICE)
        model.load_state_dict(state_dict, assign=True)

    if world_size > 1:
        with torch.no_grad():
            for param in list(model.parameters()) + list(model.buffers()):
                dist.broadcast(param, src=0)

    # Optimizer
    optimizers = [
        AdamATan2(
            model.parameters(),
            lr=0,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2)
        )
    ]
    optimizer_lrs = [config.lr]

    return model, optimizers, optimizer_lrs


def cosine_schedule_with_warmup(
    current_step: int,
    base_lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
    min_ratio: float = 0.1
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))))


def init_train_state(config: QEPretrainConfig, train_metadata: QEDatasetMetadata, rank: int, world_size: int):
    total_steps = int(config.epochs * train_metadata.total_groups / config.gradient_accumulation_steps)
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, rank=rank, world_size=world_size)

    return TrainState(
        step=0,
        total_steps=total_steps,
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None
    )


def train_batch(config: QEPretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    train_state.accum_step += 1

    batch = {k: v.to(DEVICE) for k, v in batch.items()}

    if train_state.carry is None:
        with torch.device(DEVICE):
            train_state.carry = train_state.model.initial_carry(batch)

    forward_kwargs = dict(carry=train_state.carry, batch=batch, return_keys=[])

    train_state.carry, loss, metrics, _, _ = train_state.model(**forward_kwargs)

    scaled_loss = loss / (global_batch_size * config.gradient_accumulation_steps)
    scaled_loss.backward()

    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = cosine_schedule_with_warmup(
            current_step=train_state.step,
            base_lr=base_lr,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=train_state.total_steps,
            min_ratio=config.lr_min_ratio
        )

    if train_state.accum_step >= config.gradient_accumulation_steps:
        if world_size > 1:
            for param in train_state.model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad)

        for optim in train_state.optimizers:
            for param_group in optim.param_groups:
                param_group['lr'] = lr_this_step
            optim.step()
            optim.zero_grad()

        train_state.accum_step = 0
        train_state.step += 1

        if train_state.step > train_state.total_steps:
            return

    if len(metrics):
        metric_keys = list(sorted(metrics.keys()))
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}

            count = max(reduced_metrics.get("count", 1), 1)
            final_metrics = {}

            for k, v in reduced_metrics.items():
                if k.endswith("loss") or k == "mae" or k == "rmse_sq":
                    final_metrics[f"train/{k}"] = v / global_batch_size
                elif k.startswith("lang") or k.startswith("pearson"):
                    final_metrics[f"train/{k}"] = v
                else:
                    final_metrics[f"train/{k}"] = v / count

            final_metrics["train/lr"] = lr_this_step
            return final_metrics


def evaluate(config: QEPretrainConfig, train_state: TrainState, eval_loader, eval_metadata, rank: int, world_size: int):
    """Evaluate QE model and compute Pearson/Spearman correlations."""
    all_preds = []
    all_targets = []
    all_lang_ids = []

    with torch.inference_mode():
        carry = None
        for set_name, batch, global_batch_size in eval_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            with torch.device(DEVICE):
                carry = train_state.model.initial_carry(batch)

            while True:
                carry, loss, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=["predicted_scores"]
                )
                if all_finish:
                    break

            if "predicted_scores" in preds:
                all_preds.append(preds["predicted_scores"].cpu())
            all_targets.append(batch["scores"].cpu())
            if "lang_pair_ids" in batch:
                all_lang_ids.append(batch["lang_pair_ids"].cpu())

    if not all_preds:
        return None

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_lang_ids = torch.cat(all_lang_ids).numpy() if all_lang_ids else None

    pearson, _ = stats.pearsonr(all_preds, all_targets)
    spearman, _ = stats.spearmanr(all_preds, all_targets)
    mae = np.abs(all_preds - all_targets).mean()
    rmse = np.sqrt(((all_preds - all_targets) ** 2).mean())

    metrics = {
        "eval/pearson": pearson,
        "eval/spearman": spearman,
        "eval/mae": mae,
        "eval/rmse": rmse,
    }

    # Per-language metrics
    if all_lang_ids is not None:
        for lang_id in np.unique(all_lang_ids):
            mask = all_lang_ids == lang_id
            if mask.sum() > 10:
                lp_pearson, _ = stats.pearsonr(all_preds[mask], all_targets[mask])
                lp_spearman, _ = stats.spearmanr(all_preds[mask], all_targets[mask])
                lp_mae = np.abs(all_preds[mask] - all_targets[mask]).mean()
                metrics[f"eval/lang{int(lang_id)}_pearson"] = lp_pearson
                metrics[f"eval/lang{int(lang_id)}_spearman"] = lp_spearman
                metrics[f"eval/lang{int(lang_id)}_mae"] = lp_mae

    if rank == 0:
        print(f"Eval: Pearson={pearson:.4f}, Spearman={spearman:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

    return metrics


def save_train_state(config: QEPretrainConfig, train_state: TrainState):
    if config.checkpoint_path is None:
        return
    os.makedirs(config.checkpoint_path, exist_ok=True)

    if config.checkpoint_keep_latest_only:
        ckpt_path = os.path.join(config.checkpoint_path, "latest")
        torch.save(train_state.model.state_dict(), ckpt_path)
        with open(os.path.join(config.checkpoint_path, "latest_step.txt"), "w") as f:
            f.write(str(train_state.step))
    else:
        torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}"))


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> QEPretrainConfig:
    objects = [None]
    if rank == 0:
        config = QEPretrainConfig(**hydra_config)

        if config.project_name is None:
            config.project_name = "TRM-QE"
        if config.run_name is None:
            config.run_name = f"qe-{coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]


@hydra.main(config_path="../config", config_name="base", version_base=None)
def launch(hydra_config: DictConfig):
    global DEVICE

    RANK = 0
    WORLD_SIZE = 1

    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Set global device
    DEVICE = get_device(config.device)
    print(f"Using device: {DEVICE}")

    # Set device for layers.py PretrainedEncoder
    os.environ["TRM_DEVICE"] = str(DEVICE)

    torch.random.manual_seed(config.seed + RANK)

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    train_loader, train_metadata = create_qe_dataloader(config, "train", RANK, WORLD_SIZE, train_epochs_per_iter)

    try:
        eval_loader, eval_metadata = create_qe_dataloader(config, "test", RANK, WORLD_SIZE, 1)
    except:
        print("No eval data found")
        eval_loader = eval_metadata = None

    # Train state
    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)

    # Progress bar and logger
    progress_bar = None
    ema_helper = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config=config.model_dump(),
            tags=config.wandb_tags if config.wandb_tags else None
        )
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)

    if config.ema:
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    # Training Loop
    for _iter_id in range(total_iters):
        if RANK == 0:
            print(f"Epoch {_iter_id * train_epochs_per_iter}")

        train_state.model.train()
        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)

            if config.ema:
                ema_helper.update(train_state.model)

        # Evaluation
        if eval_loader is not None and _iter_id >= config.min_eval_interval:
            if RANK == 0:
                print("EVALUATE")

            if config.ema:
                train_state_eval = copy.deepcopy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            else:
                train_state_eval = train_state

            train_state_eval.model.eval()
            metrics = evaluate(config, train_state_eval, eval_loader, eval_metadata, rank=RANK, world_size=WORLD_SIZE)

            if RANK == 0 and metrics:
                wandb.log(metrics, step=train_state.step)

            if RANK == 0 and config.checkpoint_every_eval:
                save_train_state(config, train_state_eval)

            if config.ema:
                del train_state_eval

    # Final checkpoint
    if RANK == 0:
        save_train_state(config, train_state)

    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
