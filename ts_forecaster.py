"""
ts_forecaster.py  (v3 – true variation in predictions)
=======================================================

Root cause of flat predictions (fixed here)
--------------------------------------------
The previous version used MinMaxScaler on *differences*.  That compresses
diffs into [0, 1], squashing their std from ~17 down to ~0.18.  A model
trained on near-constant targets learns to predict the mean of [0,1] ≈ 0.54,
which inverse-transforms to a nearly-zero diff every step → flat output.

Fix: StandardScaler on differences
------------------------------------
  StandardScaler always produces mean=0, std=1, regardless of the input range.
  The model then receives a properly-shaped distribution, and predicting the
  mean (0) maps back to "no change" — not to a fixed absolute level.
  This is the only reliable scaler for *differenced* time series.

Architecture changes
---------------------
  * `shuffle=False` — temporal order must be preserved
  * No augmentation — it adds random offsets that mislead a diff-based model
  * Smaller, regularised network: `[32, 16]` units + L2 weight decay
  * Validation done with a held-out *tail* slice, not random val_split,
    so the model never sees future data during training.

Public API (unchanged)
-----------------------
  model, history = train(series, params)
  predictions    = predict(model, series, params)
  forecast       = forecast_future(model, series, params, n_steps)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_windows(values: np.ndarray, look_back: int):
    """
    Slide a window of `look_back` over 1-D `values` and return (X, y).

    Parameters
    ----------
    values    : 1-D array, already scaled
    look_back : int

    Returns
    -------
    X : (n_windows, look_back, 1)   – ready for LSTM/GRU input
    y : (n_windows,)
    """
    X, y = [], []
    for i in range(len(values) - look_back):
        X.append(values[i : i + look_back])
        y.append(values[i + look_back])
    return (
        np.array(X, dtype=np.float32).reshape(-1, look_back, 1),
        np.array(y, dtype=np.float32),
    )


def _build_model(look_back: int, params: dict) -> keras.Model:
    """
    Compact LSTM / GRU with L2 regularisation.

    Parameters read from `params`
    ------------------------------
    cell_type   : 'LSTM' | 'GRU'           default 'LSTM'
    units       : list[int]                 default [32, 16]
    dropout     : float  (input dropout)    default 0.1
    dense_units : list[int]                 default [16]
    l2          : float  L2 weight penalty  default 1e-4
    """
    cell_type   = params.get("cell_type",   "LSTM").upper()
    units       = params.get("units",       [32, 16])
    dropout     = params.get("dropout",     0.1)
    dense_units = params.get("dense_units", [16])
    l2_reg      = params.get("l2",          1e-4)

    RNNCell = layers.LSTM if cell_type == "LSTM" else layers.GRU
    reg     = regularizers.l2(l2_reg)

    model = keras.Sequential(name=f"{cell_type}_v3")
    model.add(keras.Input(shape=(look_back, 1)))

    for idx, n in enumerate(units):
        return_seq = idx < len(units) - 1
        model.add(
            RNNCell(
                n,
                return_sequences=return_seq,
                dropout=dropout,
                recurrent_dropout=0.0,          # keep recurrent path clean
                kernel_regularizer=reg,
                name=f"{cell_type.lower()}_{idx+1}",
            )
        )

    for idx, d in enumerate(dense_units):
        model.add(
            layers.Dense(d, activation="relu",
                         kernel_regularizer=reg,
                         name=f"dense_{idx+1}")
        )

    # Linear activation — differences are centred around 0 and can be negative
    model.add(layers.Dense(1, activation="linear", name="output"))
    return model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train(series: pd.Series, params: dict):
    """
    Fit an LSTM/GRU model on a univariate time series.

    Strategy
    --------
    1. First-difference the series  →  removes level/trend,
       forces the model to predict *movement* not *level*.
    2. StandardScaler on the diffs  →  mean=0, std=1, preserving shape.
       (MinMaxScaler squashes diff distributions and causes flat output.)
    3. Temporal validation split    →  last `val_split` fraction held out
       in order; no shuffling of data.
    4. MAE loss                     →  does not reward mean-prediction.
    5. L2 regularisation            →  controls capacity without dropout
       destroying the signal on moderate datasets.

    Parameters
    ----------
    series : pd.Series
        Raw (unscaled) univariate time series.
    params : dict
        ┌──────────────────┬──────────────────────────────────────────────────┐
        │ Key              │ Description / default                            │
        ├──────────────────┼──────────────────────────────────────────────────┤
        │ look_back        │ Window length in time-steps          (default 7) │
        │ cell_type        │ 'LSTM' or 'GRU'                      (default LSTM)│
        │ units            │ Recurrent layer sizes                ([32, 16])  │
        │ dropout          │ Input dropout                         (0.1)      │
        │ dense_units      │ Dense hidden layer sizes              ([16])     │
        │ l2               │ L2 regularisation strength            (1e-4)     │
        │ epochs           │ Maximum epochs                        (300)      │
        │ batch_size       │ Mini-batch size                       (16)       │
        │ val_split        │ Tail fraction used for validation     (0.15)     │
        │ learning_rate    │ Adam initial LR                       (3e-4)     │
        │ patience         │ Early-stopping patience               (25)       │
        │ verbose          │ Keras verbosity 0/1/2                 (1)        │
        └──────────────────┴──────────────────────────────────────────────────┘

    Returns
    -------
    model   : keras.Model   (scaler and metadata stored as attributes)
    history : keras History object
    """
    look_back  = params.get("look_back",     7)
    epochs     = params.get("epochs",        300)
    batch_size = params.get("batch_size",    16)
    val_frac   = params.get("val_split",     0.15)
    lr         = params.get("learning_rate", 3e-4)
    patience   = params.get("patience",      25)
    verbose    = params.get("verbose",       1)

    raw = series.values.astype(np.float32)

    # ── Step 1: first differences ─────────────────────────────────────────
    # diffs[i] = raw[i+1] - raw[i]
    # This removes the absolute level and any linear trend.
    diffs = np.diff(raw)                          # shape: (n-1,)

    # ── Step 2: StandardScaler ────────────────────────────────────────────
    # Produces mean=0, std=1.  Distribution shape is preserved.
    # Crucially, predicting 0 → "no change", not a flat absolute level.
    scaler = StandardScaler()
    sdiffs = scaler.fit_transform(diffs.reshape(-1, 1)).flatten()

    # ── Step 3: supervised windows ────────────────────────────────────────
    X, y = _make_windows(sdiffs, look_back)       # X: (n_win, look_back, 1)

    # ── Step 4: temporal train/val split ─────────────────────────────────
    # Use the TAIL for validation — never let the model see future data.
    n_val  = max(1, int(len(X) * val_frac))
    n_train = len(X) - n_val
    X_train, y_train = X[:n_train], y[:n_train]
    X_val,   y_val   = X[n_train:], y[n_train:]

    if verbose:
        print(f"[train] {n_train} train windows, {n_val} val windows")
        print(f"[train] diff std={diffs.std():.3f}  scaled std={sdiffs.std():.3f}")

    # ── Step 5: build & compile ───────────────────────────────────────────
    model = _build_model(look_back, params)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mae",           # MAE: no incentive to predict the mean
        metrics=["mse"],
    )
    if verbose:
        model.summary()

    # ── Step 6: callbacks ─────────────────────────────────────────────────
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=verbose,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience // 3,
            min_lr=1e-7,
            verbose=verbose,
        ),
    ]

    # ── Step 7: train ─────────────────────────────────────────────────────
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=cb_list,
        shuffle=False,        # MUST be False — temporal order matters
        verbose=verbose,
    )

    # Attach metadata so predict/forecast work without extra arguments
    model.scaler_    = scaler      # StandardScaler fitted on diffs
    model.look_back_ = look_back
    model.raw_values_ = raw        # needed to anchor diff reconstruction

    return model, history


def predict(model: keras.Model, series: pd.Series, params: dict = None) -> pd.Series:
    """
    In-sample one-step-ahead predictions in original units.

    The first `look_back + 1` positions are NaN (no window available).

    Returns
    -------
    pd.Series  aligned with `series.index`, same units as input.
    """
    look_back = model.look_back_
    scaler    = model.scaler_

    raw    = series.values.astype(np.float32)
    diffs  = np.diff(raw)
    sdiffs = scaler.transform(diffs.reshape(-1, 1)).flatten()

    X, _ = _make_windows(sdiffs, look_back)

    # Predict scaled diffs, then invert scaling
    pred_sdiffs = model.predict(X, verbose=0).flatten()           # (n_win,)
    pred_diffs  = scaler.inverse_transform(
        pred_sdiffs.reshape(-1, 1)
    ).flatten()

    # Reconstruct absolute values:
    #   window i predicts diff[i + look_back]  → value[i + look_back + 1]
    #   prior actual value = raw[i + look_back]
    n_win  = len(pred_diffs)
    priors = raw[look_back : look_back + n_win]   # actual prior values
    preds  = priors + pred_diffs

    # Align with original index (offset = look_back + 1)
    offset     = look_back + 1
    full_preds = np.full(len(series), np.nan, dtype=np.float32)
    full_preds[offset : offset + n_win] = preds

    return pd.Series(full_preds, index=series.index, name="predicted")


def forecast_future(
    model: keras.Model,
    series: pd.Series,
    params: dict,
    n_steps: int,
) -> pd.Series:
    """
    Autoregressive forecast `n_steps` beyond the end of `series`.

    Each step:
      1. predict next scaled diff from the rolling buffer
      2. inverse-scale the diff
      3. add to the last known absolute value
      4. push the new scaled diff into the buffer

    Returns
    -------
    pd.Series  with DatetimeIndex (daily) or RangeIndex.
    """
    look_back = model.look_back_
    scaler    = model.scaler_

    raw    = series.values.astype(np.float32)
    diffs  = np.diff(raw)
    sdiffs = scaler.transform(diffs.reshape(-1, 1)).flatten()

    # Seed the rolling buffer with the last `look_back` scaled diffs
    buffer      = list(sdiffs[-look_back:])
    last_actual = float(raw[-1])
    future_vals = []

    for _ in range(n_steps):
        window      = np.array(buffer[-look_back:], dtype=np.float32).reshape(1, look_back, 1)
        next_sdiff  = float(model.predict(window, verbose=0)[0, 0])
        next_diff   = float(scaler.inverse_transform([[next_sdiff]])[0, 0])
        next_val    = last_actual + next_diff

        future_vals.append(next_val)
        last_actual = next_val
        buffer.append(next_sdiff)

    future_arr = np.array(future_vals, dtype=np.float32)

    # Build future index
    if isinstance(series.index, pd.DatetimeIndex):
        freq  = pd.infer_freq(series.index) or "D"
        start = series.index[-1] + pd.tseries.frequencies.to_offset(freq)
        idx   = pd.date_range(start=start, periods=n_steps, freq=freq)
    else:
        s = int(series.index[-1]) + 1
        idx = pd.RangeIndex(start=s, stop=s + n_steps)

    return pd.Series(future_arr, index=idx, name="forecast")
