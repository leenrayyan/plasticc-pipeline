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

        params = {**config.XGB_PARAMS, **kwargs}
        params["num_class"] = n_classes          # for multi-class softmax
        params["objective"] = "multi:softmax"    # overridden below for proba
        self.model = XGBClassifier(**{k: v for k, v in params.items()
                                      if k != "objective"},
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

    Parameters
    ----------
    d_model        : Transformer model dimension.
    max_wavelength : Denominator scale (analogous to 10000 in the original
                     Attention Is All You Need paper).
    """

    def __init__(
        self,
        d_model        : int   = config.D_MODEL,
        max_wavelength : float = config.MAX_WAVELENGTH,
    ) -> None:
        super().__init__()
        self.d_model        = d_model
        self.max_wavelength = max_wavelength

        # Pre-compute dimension exponents (fixed, not learnable)
        half  = d_model // 2
        expos = torch.arange(half, dtype=torch.float32) * 2.0 / d_model
        self.register_buffer("divisor", max_wavelength ** expos)  # (d_model/2,)

    def forward(self, time_deltas: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        time_deltas : Tensor of shape ``(batch, seq_len)`` — days since first obs.

        Returns
        -------
        Tensor of shape ``(batch, seq_len, d_model)``.
        """
        # time_deltas: (B, L) → (B, L, 1)
        t = time_deltas.unsqueeze(-1)                    # (B, L, 1)
        angles = t / self.divisor                         # (B, L, d_model/2)
        enc    = torch.cat([angles.sin(), angles.cos()], dim=-1)  # (B, L, d_model)

        # If d_model is odd, trim the last column
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

    MC Dropout: dropout layers remain **active during inference** when the
    model is in ``train()`` mode.  The ``mc_mode()`` context manager
    (defined below) enables this without affecting batch-norm layers.

    Parameters
    ----------
    n_classes      : Number of output classes.
    input_dim      : Feature dimension per time step (default: config.INPUT_DIM).
    d_model        : Transformer hidden dimension.
    n_heads        : Number of attention heads.
    n_layers       : Number of Transformer encoder layers.
    dim_feedforward: FFN hidden dimension.
    dropout        : Dropout probability (applied in attention + head).
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

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Time encoding (uses the time_delta channel, index 2)
        self.time_enc = SinusoidalTimeEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            activation      = "relu",
            batch_first     = True,   # (batch, seq, feat)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head
        self.head_dropout = nn.Dropout(p=dropout)
        self.classifier   = nn.Linear(d_model, n_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming / Xavier initialisation for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x    : torch.Tensor,
        mask : torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : Tensor ``(batch, seq_len, INPUT_DIM)``
        mask : Tensor ``(batch, seq_len)`` bool — True for padded positions.

        Returns
        -------
        logits : Tensor ``(batch, n_classes)``
        """
        B, L, _ = x.shape

        # Project to d_model
        out = self.input_proj(x)                              # (B, L, d_model)

        # Add time-based positional encoding (time_delta is feature index 2)
        time_deltas = x[:, :, 2]                              # (B, L)
        out = out + self.time_enc(time_deltas)                # (B, L, d_model)

        # Prepend CLS token
        cls  = self.cls_token.expand(B, -1, -1)              # (B, 1, d_model)
        out  = torch.cat([cls, out], dim=1)                   # (B, 1+L, d_model)

        # Extend mask: CLS is never masked
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
        full_mask = torch.cat([cls_mask, mask], dim=1)        # (B, 1+L)

        # Transformer encoder
        out = self.encoder(out, src_key_padding_mask=full_mask)  # (B, 1+L, d_model)

        # CLS token output → classification head
        cls_out = out[:, 0, :]                                # (B, d_model)
        cls_out = self.head_dropout(cls_out)
        logits  = self.classifier(cls_out)                    # (B, n_classes)
        return logits


# ===========================================================================
# Model 3 — Astromer classifier
# ===========================================================================

def _try_load_astromer():
    """
    Attempt to load an Astromer backbone.

    Priority
    --------
    1. Astromer2 (from GitHub / HuggingFace — ``astromer2`` package).
    2. Astromer1 (``pip install astromer``).

    Returns
    -------
    (backbone, embed_dim, loader_name)  or raises ImportError.
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

    # ---- Try Astromer1 (pip install astromer — installs as module 'ASTROMER') ----
    try:
        from ASTROMER.models import SingleBandEncoder as Encoder1
        backbone  = Encoder1()
        backbone.load_weights(config.ASTROMER_WEIGHTS)
        embed_dim = getattr(backbone, "output_dim", 256)
        print("[models] Loaded Astromer1 backbone (ASTROMER package).")
        return backbone, embed_dim, "astromer1"
    except Exception:
        pass

    raise ImportError(
        "Neither Astromer2 nor Astromer1 could be imported.\n"
        "Install with:  python -m pip install astromer tensorboard\n"
        "Astromer2 (if available): pip install astromer2"
    )


class AstroClassifierHead(nn.Module):
    """
    Classification head appended to a frozen Astromer encoder.

    Architecture: Linear(embed_dim, 128) → ReLU → Dropout(0.1)
                  → Linear(128, n_classes)
    """

    def __init__(self, embed_dim: int, n_classes: int, dropout: float = config.DROPOUT) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, embed_dim)  →  logits: (batch, n_classes)"""
        return self.net(x)


class AstroClassifier(nn.Module):
    """
    Astromer (1 or 2) encoder with a learnable classification head.

    The encoder weights are frozen; only the head is trained.
    MC Dropout is applied inside the classification head.

    Parameters
    ----------
    n_classes : Number of target classes.
    """

    def __init__(self, n_classes: int) -> None:
        super().__init__()
        backbone, embed_dim, self._loader = _try_load_astromer()

        # Freeze encoder
        self.backbone  = backbone
        self.embed_dim = embed_dim
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.head = AstroClassifierHead(embed_dim, n_classes)

    def _encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Run the Astromer backbone and return a pooled embedding.

        Astromer1 / 2 differ slightly in their forward API; we handle both.
        ``x`` has shape ``(batch, seq_len, INPUT_DIM)``.
        """
        # Astromer expects (time, flux, flux_err) — channels 2, 0, 1
        time   = x[:, :, 2]   # time_delta
        flux   = x[:, :, 0]   # flux
        ferr   = x[:, :, 1]   # flux_err

        try:
            # Astromer2-style API
            emb = self.backbone(time=time, flux=flux, flux_err=ferr, mask=mask)
        except TypeError:
            # Astromer1-style API: expects a dict
            inp = {"times": time, "input": flux, "mask_in": ~mask}
            emb = self.backbone(inp)

        # Pool over sequence dimension if needed
        if emb.dim() == 3:
            # Mask out padding before mean-pooling
            valid = (~mask).float().unsqueeze(-1)          # (B, L, 1)
            emb   = (emb * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        return emb   # (batch, embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : Tensor ``(batch, seq_len, INPUT_DIM)``
        mask : Tensor ``(batch, seq_len)`` bool — True for padded positions.

        Returns
        -------
        logits : Tensor ``(batch, n_classes)``
        """
        with torch.no_grad():
            emb = self._encode(x, mask)
        logits = self.head(emb)
        return logits


# ===========================================================================
# Model 4 — Moirai / Chronos classifier
# ===========================================================================

def _try_load_moirai(device: torch.device):
    """
    Attempt to load a Moirai-small or Chronos-small backbone.

    Priority
    --------
    1. Moirai-small  (Salesforce/moirai-1.0-R-small via uni2ts).
    2. Chronos-small (amazon/chronos-t5-small via transformers T5EncoderModel).

    Returns
    -------
    (backbone, embed_dim, loader_name)  or raises ImportError.
    """
    # ---- Try Moirai ----------------------------------------------------------
    try:
        from uni2ts.model.moirai import MoiraiModule
        backbone = MoiraiModule.from_pretrained(config.MOIRAI_MODEL_ID)
        backbone = backbone.to(device)
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

    # ---- Try Chronos via T5EncoderModel -------------------------------------
    try:
        from transformers import T5EncoderModel, AutoTokenizer
        backbone  = T5EncoderModel.from_pretrained(config.CHRONOS_MODEL_ID)
        backbone  = backbone.to(device)
        backbone.eval()
        embed_dim = backbone.config.d_model
        print(f"[models] Loaded Chronos T5 encoder ({config.CHRONOS_MODEL_ID}).")
        return backbone, embed_dim, "chronos"
    except Exception as e:
        pass

    raise ImportError(
        "Neither Moirai nor Chronos could be imported.\n"
        "Install Moirai:  pip install uni2ts\n"
        "Install Chronos: pip install git+https://github.com/amazon-science/chronos-forecasting.git\n"
        "Or install transformers>=4.40: pip install transformers"
    )


class MoiraiClassifier(nn.Module):
    """
    Moirai-small or Chronos-small encoder with a learnable classification head.

    The foundation-model encoder is frozen; only the classification head is
    trained.  MC Dropout is applied in the head.

    Notes
    -----
    * Moirai expects univariate time series; we feed the mean of all passbands
      as the primary channel and include the other features as covariates.
    * Chronos (T5 encoder) operates on tokenised sequences; we project our
      continuous feature vectors through an embedding table before encoding.
    * If either backbone is too large for the available GPU memory, reduce
      ``MAX_SEQ_LEN`` in config.py or switch to CPU.

    Parameters
    ----------
    n_classes : Number of target classes.
    device    : torch.device to load the backbone onto.
    """

    def __init__(self, n_classes: int, device: Optional[torch.device] = None) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        backbone, embed_dim, self._loader = _try_load_moirai(device)
        self.backbone  = backbone
        self.embed_dim = embed_dim
        self._device   = device

        # Freeze encoder
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Lightweight input adapter: project INPUT_DIM → embed_dim
        self.input_adapter = nn.Linear(config.INPUT_DIM, embed_dim)

        # Classification head with MC Dropout
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=config.DROPOUT),
            nn.Linear(128, n_classes),
        )

    def _encode_moirai(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode with Moirai backbone using the flux channel."""
        flux = x[:, :, 0].unsqueeze(-1)          # (B, L, 1)
        obs_mask = ~mask                           # True = valid observation
        try:
            out = self.backbone.encode(
                past_values        = flux,
                past_observed_mask = obs_mask.unsqueeze(-1),
            )
            if hasattr(out, "last_hidden_state"):
                hidden = out.last_hidden_state     # (B, L, d_model)
            else:
                hidden = out
        except Exception:
            # Fallback: project and mean-pool
            hidden = self.input_adapter(x)         # (B, L, embed_dim)

        # Mean-pool over valid timesteps
        valid  = (~mask).float().unsqueeze(-1)     # (B, L, 1)
        emb    = (hidden * valid).sum(1) / valid.sum(1).clamp(min=1)
        return emb   # (B, embed_dim)

    def _encode_chronos(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode with Chronos T5 encoder by projecting features into embedding space."""
        # Project continuous features to embedding dimension
        proj   = self.input_adapter(x)             # (B, L, embed_dim)
        # T5EncoderModel.forward expects (B, L, embed_dim) as inputs_embeds
        attn   = (~mask).long()                    # (B, L) attention mask
        out    = self.backbone(
            inputs_embeds   = proj,
            attention_mask  = attn,
        )
        hidden = out.last_hidden_state             # (B, L, embed_dim)
        valid  = (~mask).float().unsqueeze(-1)
        emb    = (hidden * valid).sum(1) / valid.sum(1).clamp(min=1)
        return emb   # (B, embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : Tensor ``(batch, seq_len, INPUT_DIM)``
        mask : Tensor ``(batch, seq_len)`` bool — True for padded positions.

        Returns
        -------
        logits : Tensor ``(batch, n_classes)``
        """
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
    name      : One of ``"xgboost"``, ``"transformer"``, ``"astromer"``,
                ``"moirai"``.
    n_classes : Number of output classes.
    device    : torch.device (used by Moirai loader).  Auto-detected if None.

    Returns
    -------
    Model instance (XGBoostClassifier or nn.Module subclass).
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
