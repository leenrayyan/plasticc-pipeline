"""
models.py — All four classifier architectures.

Model 1  XGBoostClassifier   sklearn-compatible wrapper around XGBoost,
                              trained on hand-crafted features from features.py.

Model 2  TransformerClassifier
                              Scratch Transformer with:
                              - Sinusoidal positional encoding using *actual*
                                time-delta values (not sequence indices).
                              - CLS-token classification head.
                              - MC Dropout (p=0.1) active during inference.

Model 3  AstroClassifier      Pretrained Astromer2 / Astromer1 encoder
                              (frozen) + learnable classification head with
                              MC Dropout.

Model 4  MoiraiClassifier     Moirai-small (Salesforce) or Chronos-small
                              (Amazon, fallback) encoder + classification
                              head with MC Dropout.

Factory  build_model(name, n_classes, …) → model instance.
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # Must be set before any TF/Keras import

import math
import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ===========================================================================
# Model 1 — XGBoost wrapper
# ===========================================================================

class XGBoostClassifier:
    """
    Thin sklearn-compatible wrapper around ``xgboost.XGBClassifier``.

    Accepts ``numpy`` arrays for training; returns ``numpy`` probability
    matrices from ``predict_proba``.
    """

    def __init__(self, n_classes: int, **kwargs) -> None:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for Model 1. Install it with: pip install xgboost"
            ) from exc

        # Remove objective/num_class from params to avoid duplicate keyword errors,
        # then pass them explicitly with the correct values.
        params = {k: v for k, v in {**config.XGB_PARAMS, **kwargs}.items()
                  if k not in ("objective", "num_class")}
        self.model = XGBClassifier(**params,
                                   objective="multi:softprob",
                                   num_class=n_classes)
        self.n_classes = n_classes
        self.is_fitted  = False

    def fit(
        self,
        X_train : np.ndarray,
        y_train : np.ndarray,
        X_val   : Optional[np.ndarray] = None,
        y_val   : Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "XGBoostClassifier":
        """Fit the XGBoost model on ``(X_train, y_train)``."""
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(
            X_train, y_train,
            eval_set            = eval_set,
            sample_weight       = sample_weight,
            verbose             = False,
        )
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability matrix of shape ``(n_samples, n_classes)``."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict_proba().")
        probs = self.model.predict_proba(X)
        return probs.astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices of shape ``(n_samples,)``."""
        return np.argmax(self.predict_proba(X), axis=1)

    def save(self, path: str) -> None:
        """Persist the fitted model to *path* (JSON format)."""
        self.model.save_model(path)

    def load(self, path: str) -> None:
        """Load a previously saved model from *path*."""
        self.model.load_model(path)
        self.is_fitted = True


# ===========================================================================
# Model 2 — Transformer (from scratch)
# ===========================================================================

class SinusoidalTimeEncoding(nn.Module):
    """
    Sinusoidal positional encoding computed from *actual* time-delta values.

    Unlike the standard formulation (which uses integer position indices),
    here the encoding is a function of the continuous observation time::

        PE(t, 2i)   = sin( t / max_wavelength^(2i / d_model) )
        PE(t, 2i+1) = cos( t / max_wavelength^(2i / d_model) )

    This preserves the irregular-cadence structure of the light curves.
    """

    def __init__(
        self,
        d_model        : int   = config.D_MODEL,
        max_wavelength : float = config.MAX_WAVELENGTH,
    ) -> None:
        super().__init__()
        self.d_model        = d_model
        self.max_wavelength = max_wavelength

        half  = d_model // 2
        expos = torch.arange(half, dtype=torch.float32) * 2.0 / d_model
        self.register_buffer("divisor", max_wavelength ** expos)

    def forward(self, time_deltas: torch.Tensor) -> torch.Tensor:
        t      = time_deltas.unsqueeze(-1)
        angles = t / self.divisor
        enc    = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return enc[..., : self.d_model]


class TransformerClassifier(nn.Module):
    """
    Transformer-based transient classifier trained from scratch.

    Architecture
    ------------
    1. Linear input projection: INPUT_DIM → d_model.
    2. Prepend a learnable CLS token.
    3. Add sinusoidal time encoding (time_delta from feature dim 2).
    4. N encoder layers (multi-head self-attention + FFN).
    5. Extract CLS output → Dropout → Linear(d_model, n_classes).

    MC Dropout: dropout layers remain active during inference.
    """

    def __init__(
        self,
        n_classes       : int,
        input_dim       : int   = config.INPUT_DIM,
        d_model         : int   = config.D_MODEL,
        n_heads         : int   = config.N_HEADS,
        n_layers        : int   = config.N_ENCODER_LAYERS,
        dim_feedforward : int   = config.DIM_FEEDFORWARD,
        dropout         : float = config.DROPOUT,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(input_dim, d_model)

        # Layer norm stabilises activations before encoder, preventing NaN during LR warmup
        self.input_norm = nn.LayerNorm(d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.time_enc = SinusoidalTimeEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            activation      = "relu",
            batch_first     = True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head_dropout = nn.Dropout(p=dropout)
        self.classifier   = nn.Linear(d_model, n_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape

        # Sanitize inputs — NaN/Inf can appear in padded positions
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        out = self.input_proj(x)

        # Clamp time deltas to prevent sinusoidal overflow
        time_deltas = x[:, :, 2].clamp(min=0.0, max=1000.0)
        out = out + self.time_enc(time_deltas)

        # Normalise before encoder to prevent gradient explosion
        out = self.input_norm(out)

        cls       = self.cls_token.expand(B, -1, -1)
        out       = torch.cat([cls, out], dim=1)

        cls_mask  = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
        full_mask = torch.cat([cls_mask, mask], dim=1)

        out     = self.encoder(out, src_key_padding_mask=full_mask)
        cls_out = out[:, 0, :]
        cls_out = self.head_dropout(cls_out)
        logits  = self.classifier(cls_out)
        return logits


# ===========================================================================
# Model 3 — Astromer classifier
# ===========================================================================

def _try_load_astromer():
    """
    Attempt to load an Astromer backbone.

    Priority
    --------
    1. Astromer2 (astromer2 package).
    2. Astromer1 (pip install astromer).
    """
    # ---- Try Astromer2 -------------------------------------------------------
    try:
        from astromer2 import SingleBandEncoder as Encoder2
        backbone  = Encoder2.from_pretrained("astromer2-base")
        embed_dim = backbone.config.hidden_size
        print("[models] Loaded Astromer2 backbone.")
        return backbone, embed_dim, "astromer2"
    except Exception:
        pass

    # ---- Try Astromer1 -------------------------------------------------------
    try:
        from ASTROMER.models import SingleBandEncoder as Encoder1
        backbone  = Encoder1()

        # Load pretrained weights from MACHO survey (downloads automatically if not cached)
        backbone.from_pretraining('macho')
        print("[models] Loaded Astromer1 pretrained weights (MACHO survey).")

        embed_dim = getattr(backbone, "d_model", getattr(backbone, "output_dim", 200))
        print("[models] Loaded Astromer1 backbone (ASTROMER package).")
        return backbone, embed_dim, "astromer1"
    except Exception as e:
        pass

    raise ImportError(
        "Neither Astromer2 nor Astromer1 could be imported.\n"
        "Install with:  python -m pip install astromer tensorboard\n"
        "Also set os.environ['TF_USE_LEGACY_KERAS'] = '1' before importing.\n"
        "Astromer2 (if available): pip install astromer2"
    )


class AstroClassifierHead(nn.Module):
    """Classification head: Linear(embed_dim, 128) → ReLU → Dropout → Linear(128, n_classes)"""

    def __init__(self, embed_dim: int, n_classes: int, dropout: float = config.DROPOUT) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AstroClassifier(nn.Module):
    """
    Astromer (1 or 2) encoder with a learnable classification head.

    The encoder weights are frozen; only the head is trained.
    MC Dropout is applied inside the classification head.
    """

    def __init__(self, n_classes: int) -> None:
        super().__init__()
        backbone, embed_dim, self._loader = _try_load_astromer()

        self.backbone  = backbone
        self.embed_dim = embed_dim

        # Freeze encoder — only train the classification head
        try:
            for param in self.backbone.parameters():
                param.requires_grad = False
        except Exception:
            pass  # TF/Keras models don't use .parameters()

        self.head = AstroClassifierHead(embed_dim, n_classes)

    def _encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Run Astromer encoder layer directly and return pooled embedding."""
        import numpy as np
        import tensorflow as tf

        time    = x[:, :, 2].cpu().numpy().astype(np.float32)
        flux    = x[:, :, 0].cpu().numpy().astype(np.float32)
        mask_np = mask.cpu().numpy().astype(np.float32)  # 1=padded, 0=valid

        B, L = time.shape

        # Astromer encoder expects a dict with TF tensors
        batch = {
            "times"  : tf.constant(time.reshape(B, L, 1)),
            "input"  : tf.constant(flux.reshape(B, L, 1)),
            "mask_in": tf.constant((1 - mask_np).reshape(B, L, 1)),  # 1=valid
        }

        # Call the internal encoder layer directly
        encoder = self.backbone.model.get_layer("encoder")
        emb     = encoder(batch).numpy()  # (B, L, d_model)

        # Mean-pool over valid timesteps
        valid = (1 - mask_np).reshape(B, L, 1)
        emb   = (emb * valid).sum(axis=1) / valid.sum(axis=1).clip(min=1)
        emb   = torch.tensor(emb, dtype=torch.float32, device=x.device)
        return emb  # (B, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        emb    = self._encode(x, mask)
        logits = self.head(emb)
        return logits


# ===========================================================================
# Model 4 — Moirai / Chronos classifier
# ===========================================================================

def _try_load_moirai(device: torch.device):
    """
    Attempt to load Moirai-small or Chronos-small backbone.

    Priority: Moirai → Chronos.
    """
    # ---- Try Moirai ----------------------------------------------------------
    try:
        from uni2ts.model.moirai import MoiraiModule
        backbone  = MoiraiModule.from_pretrained(config.MOIRAI_MODEL_ID)
        backbone  = backbone.to(device)
        backbone.eval()
        embed_dim = backbone.config.d_model
        print(f"[models] Loaded Moirai backbone ({config.MOIRAI_MODEL_ID}).")
        return backbone, embed_dim, "moirai"
    except Exception as e:
        warnings.warn(
            f"[models] Moirai unavailable ({e}). "
            "Falling back to Chronos (amazon/chronos-t5-small).",
            stacklevel=2,
        )

    # ---- Try Chronos ---------------------------------------------------------
    try:
        from transformers import T5EncoderModel
        backbone  = T5EncoderModel.from_pretrained(config.CHRONOS_MODEL_ID)
        backbone  = backbone.to(device)
        backbone.eval()
        embed_dim = backbone.config.d_model
        print(f"[models] Loaded Chronos T5 encoder ({config.CHRONOS_MODEL_ID}).")
        return backbone, embed_dim, "chronos"
    except Exception:
        pass

    raise ImportError(
        "Neither Moirai nor Chronos could be imported.\n"
        "Install Moirai:  pip install uni2ts\n"
        "Or install transformers>=4.40: pip install transformers"
    )


class MoiraiClassifier(nn.Module):
    """
    Moirai-small or Chronos-small encoder with a learnable classification head.
    Encoder is frozen; only the head is trained. MC Dropout in the head.
    """

    def __init__(self, n_classes: int, device: Optional[torch.device] = None) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        backbone, embed_dim, self._loader = _try_load_moirai(device)
        self.backbone      = backbone
        self.embed_dim     = embed_dim
        self._device       = device

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.input_adapter = nn.Linear(config.INPUT_DIM, embed_dim)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=config.DROPOUT),
            nn.Linear(128, n_classes),
        )

    def _encode_moirai(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        flux     = x[:, :, 0].unsqueeze(-1)
        obs_mask = ~mask
        try:
            out = self.backbone.encode(
                past_values        = flux,
                past_observed_mask = obs_mask.unsqueeze(-1),
            )
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
        except Exception:
            hidden = self.input_adapter(x)

        valid = (~mask).float().unsqueeze(-1)
        emb   = (hidden * valid).sum(1) / valid.sum(1).clamp(min=1)
        return emb

    def _encode_chronos(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        proj   = self.input_adapter(x)
        attn   = (~mask).long()
        out    = self.backbone(inputs_embeds=proj, attention_mask=attn)
        hidden = out.last_hidden_state
        valid  = (~mask).float().unsqueeze(-1)
        emb    = (hidden * valid).sum(1) / valid.sum(1).clamp(min=1)
        return emb

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self._loader == "moirai":
                emb = self._encode_moirai(x, mask)
            else:
                emb = self._encode_chronos(x, mask)
        logits = self.head(emb)
        return logits


# ===========================================================================
# Factory
# ===========================================================================

def build_model(
    name      : str,
    n_classes : int,
    device    : Optional[torch.device] = None,
) -> object:
    """
    Instantiate a model by name.

    Parameters
    ----------
    name      : One of 'xgboost', 'transformer', 'astromer', 'moirai'.
    n_classes : Number of output classes.
    device    : torch.device. Auto-detected if None.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name = name.lower()
    if name == "xgboost":
        return XGBoostClassifier(n_classes=n_classes)
    elif name == "transformer":
        return TransformerClassifier(n_classes=n_classes).to(device)
    elif name == "astromer":
        return AstroClassifier(n_classes=n_classes).to(device)
    elif name == "moirai":
        return MoiraiClassifier(n_classes=n_classes, device=device)
    else:
        raise ValueError(
            f"Unknown model name {name!r}. "
            "Choose from: 'xgboost', 'transformer', 'astromer', 'moirai'."
        )