"""
Core layer implementations for TRM-QE.

Includes:
- RMSNorm
- Rotary Position Embeddings (RoPE)
- Multi-head Attention
- SwiGLU MLP
- Pretrained embedding loading
"""

from typing import Tuple, Optional
import einops
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention

from .common import trunc_normal_init_


def _get_device():
    """Get device from environment variable or auto-detect."""
    import os
    device_str = os.environ.get("TRM_DEVICE", "auto")
    if device_str and device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PretrainedEncoder(nn.Module):
    """
    Wrapper for using a full pretrained encoder (e.g., XLM-R) instead of just embeddings.
    Returns contextualized representations from the encoder.
    """

    def __init__(self, model_name: str, freeze: bool = False, output_dim: int = None):
        super().__init__()
        from transformers import AutoModel, AutoConfig

        print(f"Loading full pretrained encoder: {model_name}...")

        config = AutoConfig.from_pretrained(model_name)
        self.vocab_size = config.vocab_size
        print(f"  Vocab size: {self.vocab_size}")

        self.encoder = AutoModel.from_pretrained(model_name)

        # Get device from environment or auto-detect
        device = _get_device()
        self.encoder = self.encoder.to(device)
        print(f"  Moved encoder to {device}")

        self.hidden_size = self.encoder.config.hidden_size

        # Optional projection to different output dim
        self.projection = None
        if output_dim is not None and output_dim != self.hidden_size:
            print(f"  Adding projection: {self.hidden_size} -> {output_dim}")
            self.projection = nn.Linear(self.hidden_size, output_dim)
            self.projection = self.projection.to(device)
            self.output_dim = output_dim
        else:
            self.output_dim = self.hidden_size

        if freeze:
            print("  Freezing encoder weights")
            for param in self.encoder.parameters():
                param.requires_grad = False

        print(f"  Encoder loaded: {sum(p.numel() for p in self.encoder.parameters())/1e6:.1f}M params")
        self._debug_printed = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len] optional

        Returns:
            Contextualized embeddings: [batch, seq_len, output_dim]
        """
        device = next(self.encoder.parameters()).device
        input_ids = input_ids.to(device)

        if not self._debug_printed:
            max_id = input_ids.max().item()
            min_id = input_ids.min().item()
            print(f"  [DEBUG] First batch: shape={input_ids.shape}, min={min_id}, max={max_id}, vocab_size={self.vocab_size}, device={device}")
            self._debug_printed = True
            if max_id >= self.vocab_size or min_id < 0:
                print(f"  WARNING: Token IDs out of range!")

        input_ids = input_ids.clamp(0, self.vocab_size - 1)

        if attention_mask is None:
            attention_mask = (input_ids != 0).long()
        attention_mask = attention_mask.to(device)

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        if self.projection is not None:
            hidden_states = self.projection(hidden_states)

        return hidden_states


def load_pretrained_embeddings(model_name: str, target_dim: int) -> torch.Tensor:
    """
    Load pretrained embeddings from a HuggingFace model.
    Projects to target_dim if dimensions don't match.

    Args:
        model_name: HuggingFace model name (e.g., 'google/mt5-small')
        target_dim: Target embedding dimension

    Returns:
        Embedding weight tensor (vocab_size, target_dim)
    """
    from transformers import AutoModel

    print(f"Loading pretrained embeddings from {model_name}...")
    model = AutoModel.from_pretrained(model_name, use_safetensors=True)

    # Get embedding weights
    if hasattr(model, 'shared'):  # T5/mT5 style
        pretrained_weights = model.shared.weight.data.clone()
    elif hasattr(model, 'embeddings'):  # BERT style
        pretrained_weights = model.embeddings.word_embeddings.weight.data.clone()
    else:
        raise ValueError(f"Cannot find embeddings in {model_name}")

    vocab_size, pretrained_dim = pretrained_weights.shape
    print(f"  Pretrained: {vocab_size} tokens, {pretrained_dim} dim")

    # Project if dimensions don't match
    if pretrained_dim != target_dim:
        print(f"  Projecting {pretrained_dim} -> {target_dim} via truncated SVD")
        U, S, V = torch.svd(pretrained_weights)
        if target_dim < pretrained_dim:
            pretrained_weights = U[:, :target_dim] * S[:target_dim]
        else:
            padding = torch.randn(vocab_size, target_dim - pretrained_dim) * 0.02
            pretrained_weights = torch.cat([pretrained_weights, padding], dim=1)

    print(f"  Final: {pretrained_weights.shape}")
    del model

    return pretrained_weights


CosSin = Tuple[torch.Tensor, torch.Tensor]


def _find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary position embeddings to query and key tensors."""
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class CastedLinear(nn.Module):
    """Linear layer with automatic dtype casting."""

    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features, )))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(
            input,
            self.weight.to(input.dtype),
            bias=self.bias.to(input.dtype) if self.bias is not None else None
        )


class CastedEmbedding(nn.Module):
    """Embedding layer with optional pretrained weights and freezing."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        init_std: float,
        cast_to: torch.dtype,
        pretrained_weights: Optional[torch.Tensor] = None,
        freeze: bool = False
    ):
        super().__init__()
        self.cast_to = cast_to
        self.freeze = freeze

        if pretrained_weights is not None:
            pretrained_vocab, pretrained_dim = pretrained_weights.shape
            assert pretrained_dim == embedding_dim, \
                f"Pretrained dim {pretrained_dim} != expected {embedding_dim}"

            if pretrained_vocab != num_embeddings:
                print(f"  Adjusting pretrained vocab {pretrained_vocab} -> {num_embeddings}")
                if pretrained_vocab > num_embeddings:
                    pretrained_weights = pretrained_weights[:num_embeddings]
                else:
                    padding = torch.randn(num_embeddings - pretrained_vocab, embedding_dim) * 0.02
                    pretrained_weights = torch.cat([pretrained_weights, padding], dim=0)

            self.embedding_weight = nn.Parameter(pretrained_weights.clone())
            print(f"  Initialized embeddings from pretrained weights (freeze={freeze})")
        else:
            self.embedding_weight = nn.Parameter(
                trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
            )

        if freeze:
            self.embedding_weight.requires_grad = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    """Multi-head attention with RoPE support."""

    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        self.qkv_proj = CastedLinear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=False
        )
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)

        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        query, key, value = map(
            lambda t: einops.rearrange(t, 'B S H D -> B H S D'),
            (query, key, value)
        )
        attn_output = scaled_dot_product_attention(
            query=query, key=key, value=value, is_causal=self.causal
        )
        attn_output = einops.rearrange(attn_output, 'B H S D -> B S H D')
        attn_output = attn_output.view(batch_size, seq_len, self.output_size)

        return self.o_proj(attn_output)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    """RMS normalization."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)

    return hidden_states.to(input_dtype)
