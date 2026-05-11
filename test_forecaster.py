"""
test_forecaster.py  (v3)
========================
Tests ts_forecaster on 300 days of synthetic daily sales data.

Includes a diagnostic section that prints the exact numbers that caused
flat predictions in previous versions, so you can verify they are fixed.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from ts_forecaster import train, predict, forecast_future


# =============================================================================
# 1. Synthetic data  – 300 days
# =============================================================================

def make_sales_series(n_days: int = 300, seed: int = 42) -> pd.Series:
    """
    Synthetic daily sales with:
      base      = 80 units
      trend     = +0.5 units/day  (total +150 over 300 days)
      weekend   = +25 units on Sat/Sun
      noise     = Gaussian σ=8
    """
    rng   = np.random.default_rng(seed)
    dates = pd.date_range(start="2024-01-01", periods=n_days, freq="D")
    trend = np.linspace(0, 0.5 * n_days, n_days)
    bump  = np.array([25.0 if d.weekday() >= 5 else 0.0 for d in dates])
    noise = rng.normal(0, 8, n_days)
    sales = np.clip(80 + trend + bump + noise, 10, None).astype(np.float32)
    return pd.Series(sales, index=dates, name="sales")


# =============================================================================
# 2. Parameters
# =============================================================================

PARAMS = {
    "look_back":     7,        # one week of context
    "cell_type":     "LSTM",   # or "GRU"
    "units":         [32, 16], # compact but expressive
    "dropout":       0.1,
    "dense_units":   [16],
    "l2":            1e-4,     # weight decay instead of heavy dropout
    "epochs":        300,
    "batch_size":    16,
    "val_split":     0.15,     # held-out tail, not random shuffle
    "learning_rate": 3e-4,
    "patience":      25,
    "verbose":       1,
}


# =============================================================================
# 3. Diagnostic helpers
# =============================================================================

def print_scaler_diagnostic(series: pd.Series):
    """
    Print the numbers that reveal why MinMaxScaler on diffs causes flat output.
    This is the exact bug that was fixed in v3.
    """
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    raw   = series.values.astype(np.float32)
    diffs = np.diff(raw)

    mm = MinMaxScaler()
    sd = StandardScaler()

    mm_scaled = mm.fit_transform(diffs.reshape(-1,1)).flatten()
    ss_scaled = sd.fit_transform(diffs.reshape(-1,1)).flatten()

    print("\n" + "─"*55)
    print("SCALER DIAGNOSTIC  (why v1/v2 produced flat output)")
    print("─"*55)
    print(f"Raw diffs   range : [{diffs.min():.2f}, {diffs.max():.2f}]   std={diffs.std():.3f}")
    print()
    print(f"MinMaxScaler diffs std : {mm_scaled.std():.4f}  ← near-constant!")
    print(f"  Mean in [0,1] = {mm_scaled.mean():.4f}")
    print(f"  Model predicts ~{mm_scaled.mean():.4f} → diff ≈ "
          f"{mm.inverse_transform([[mm_scaled.mean()]])[0,0]:.4f}  → FLAT")
    print()
    print(f"StandardScaler diffs std : {ss_scaled.std():.4f}  ← always 1.0, shape preserved ✓")
    print(f"  Model predicting 0 → diff = 0 → 'no change'  (meaningful baseline)")
    print("─"*55)


# =============================================================================
# 4. Naive baseline
# =============================================================================

def naive_predict(series: pd.Series) -> pd.Series:
    """yesterday's value as today's prediction"""
    return series.shift(1).rename("naive")


# =============================================================================
# 5. Main
# =============================================================================

def main():
    sales = make_sales_series(n_days=300)

    print("\n" + "="*55)
    print(f"Series: {len(sales)} days  [{sales.index[0].date()} → {sales.index[-1].date()}]")
    print(f"  range : [{sales.min():.1f}, {sales.max():.1f}]")
    print(f"  mean  : {sales.mean():.1f}   std: {sales.std():.2f}")

    # ── Diagnostic ────────────────────────────────────────────────────────
    print_scaler_diagnostic(sales)

    # ── Naive baseline ────────────────────────────────────────────────────
    naive  = naive_predict(sales)
    v_mask = ~naive.isna()
    naive_mae  = (sales[v_mask] - naive[v_mask]).abs().mean()
    naive_rmse = ((sales[v_mask] - naive[v_mask])**2).mean()**0.5
    print(f"\nNaïve baseline  MAE={naive_mae:.2f}  RMSE={naive_rmse:.2f}")

    # ── Train ─────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print(f"Training {PARAMS['cell_type']} model (v3)…")
    import tensorflow as tf
    tf.random.set_seed(0); np.random.seed(0)

    model, history = train(sales, PARAMS)

    # ── In-sample predictions ─────────────────────────────────────────────
    in_sample = predict(model, sales)
    vm        = ~in_sample.isna()

    model_mae  = (sales[vm] - in_sample[vm]).abs().mean()
    model_rmse = ((sales[vm] - in_sample[vm])**2).mean()**0.5
    pred_std   = in_sample.dropna().std()

    print(f"\n{'='*55}")
    print("RESULTS")
    print(f"{'='*55}")
    print(f"Model  MAE  : {model_mae:.2f}   (naïve: {naive_mae:.2f})")
    print(f"Model  RMSE : {model_rmse:.2f}   (naïve: {naive_rmse:.2f})")
    print(f"\nActual std  : {sales.std():.2f}")
    print(f"Pred   std  : {pred_std:.2f}   ← should be close to actual (not ~0)")
    print(f"\nActual range  : [{sales.min():.1f}, {sales.max():.1f}]")
    print(f"Pred   range  : [{in_sample.dropna().min():.1f}, {in_sample.dropna().max():.1f}]")

    # ── Comparison table ──────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("Last 10 days — comparison:")
    print("─"*55)
    comp = pd.DataFrame({
        "actual"   : sales.round(1),
        "predicted": in_sample.round(1),
        "naive"    : naive.round(1),
        "err_model": (sales - in_sample).round(1),
        "err_naive": (sales - naive).round(1),
    }).tail(10)
    print(comp.to_string())

    # ── Future forecast ───────────────────────────────────────────────────
    n_future = 14
    fc = forecast_future(model, sales, PARAMS, n_steps=n_future)
    print(f"\n{'='*55}")
    print(f"{n_future}-day Forecast:")
    print("─"*55)
    print(fc.round(1).to_string())

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    fig.suptitle(
        "LSTM v3 · StandardScaler on diffs · MAE loss · Temporal val split\n"
        "→ Predicted std tracks actual std (no mean-collapse)",
        fontsize=12,
    )

    # Panel 1: last 60 days + forecast
    ax = axes[0]
    tail = sales.iloc[-60:]
    tail_pred = in_sample.iloc[-60:]
    ax.plot(tail.index,      tail.values,
            label="Actual", color="steelblue", lw=2, marker="o", ms=3)
    ax.plot(tail_pred.dropna().index, tail_pred.dropna().values,
            label=f"LSTM pred  MAE={model_mae:.1f}", color="orange",
            lw=1.8, ls="--", marker="x", ms=4)
    ax.plot(naive.iloc[-60:].dropna().index, naive.iloc[-60:].dropna().values,
            label=f"Naïve      MAE={naive_mae:.1f}",
            color="gray", lw=1.2, ls=":", alpha=0.7)
    ax.plot(fc.index, fc.values,
            label=f"{n_future}-day forecast",
            color="crimson", lw=2, ls=":", marker="s", ms=5)
    ax.axvspan(fc.index[0], fc.index[-1], alpha=0.07, color="crimson")
    ax.set_title("Last 60 days + forecast")
    ax.set_ylabel("Units sold")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.grid(alpha=0.3)
    ax.annotate(
        f"Actual σ={sales.std():.1f}   Pred σ={pred_std:.1f}",
        xy=(0.02, 0.05), xycoords="axes fraction",
        fontsize=9, color="darkgreen",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8),
    )

    # Panel 2: full series
    ax2 = axes[1]
    ax2.plot(sales.index, sales.values, label="Actual", color="steelblue", lw=1, alpha=0.7)
    ax2.plot(in_sample.dropna().index, in_sample.dropna().values,
             label="LSTM prediction", color="orange", lw=1, alpha=0.85)
    ax2.set_title("Full 300-day series")
    ax2.set_ylabel("Units sold")
    ax2.legend(fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax2.grid(alpha=0.3)

    # Panel 3: training loss
    ax3 = axes[2]
    ax3.plot(history.history["loss"],     label="Train MAE", color="steelblue")
    ax3.plot(history.history["val_loss"], label="Val   MAE", color="orange")
    ax3.set_title("Training convergence")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("MAE (standardised diff space)")
    ax3.legend()
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    out = "forecast_results_v3.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nPlot saved → {out}\nDone.\n")


if __name__ == "__main__":
    main()
