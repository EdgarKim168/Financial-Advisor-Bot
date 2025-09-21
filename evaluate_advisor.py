# Evaluation script for "Financial Advisor Bot"

import argparse
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up, down = d.clip(lower=0), -d.clip(upper=0)
    rs = up.rolling(n).mean() / (down.rolling(n).mean().replace(0, np.nan))
    return (100 - 100 / (1 + rs)).fillna(50)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean().fillna(method="bfill")


def make_recommendation(row, risk_mult_sl=1.5, risk_mult_tp=2.5):
    def f(v):
        if hasattr(v, "values"):
            v = np.asarray(v).ravel()[-1]
        return float(v)

    sma20, sma50 = f(row["SMA20"]), f(row["SMA50"])
    rsi14, atr14  = f(row["RSI14"]), f(row["ATR14"])
    price         = f(row["Close"])

    signal, reason = "HOLD", []
    if (sma20 > sma50) and (rsi14 < 70):
        signal, reason = "BUY", ["SMA20>SMA50, RSI<70"]
    elif (sma20 < sma50) and (rsi14 > 30):
        signal, reason = "SELL", ["SMA20<SMA50, RSI>30"]

    if signal == "BUY":
        sl, tp = price - risk_mult_sl * atr14, price + risk_mult_tp * atr14
        conf = min(0.95, 0.55 + (sma20 - sma50) / (0.5 * atr14 + 1e-9))
    elif signal == "SELL":
        sl, tp = price + risk_mult_sl * atr14, price - risk_mult_tp * atr14
        conf = min(0.95, 0.55 + (sma50 - sma20) / (0.5 * atr14 + 1e-9))
    else:
        sl = tp = np.nan
        conf = 0.45

    return dict(signal=signal, price=price, sl=sl, tp=tp,
                rsi=rsi14, atr=atr14, confidence=float(np.clip(conf, 0.05, 0.95)),
                rationale="; ".join(reason) if reason else "Mixed")


def simulate_trades(df_feat: pd.DataFrame,
                    interval_code: str,
                    spread_pips: float = 1.0,
                    slip_pips: float = 0.5,
                    timeout_bars: int = 200,
                    risk_frac: float = 0.01):
    """
    Walk forward bar by bar:
    - when flat, if signal is BUY/SELL, open at next bar open +/- costs
    - close on first touch of SL/TP; else timeout
    Returns:
      trades DataFrame,
      equity (account-style, compounding with risk_frac),
      equity_R (cumulative R, starting at 0).
    """
    feats = df_feat.copy()

    trades = []
    pos = None  
    entry_idx = None

    for i in range(len(feats) - 1):  
        row = feats.iloc[i]
        next_open = float(feats["Open"].iloc[i + 1])
        sig = make_recommendation(row)

        if pos is None and sig["signal"] in ("BUY", "SELL") and np.isfinite(sig["sl"]) and np.isfinite(sig["tp"]):
            cost = 0.0001 * (spread_pips + slip_pips)
            if sig["signal"] == "BUY":
                entry = next_open + cost
            else:
                entry = next_open - cost
            pos = dict(side=sig["signal"],
                       entry=entry,
                       sl=float(sig["sl"]),
                       tp=float(sig["tp"]),
                       conf=float(sig["confidence"]))
            entry_idx = i + 1
            continue

        if pos is not None:
            high = float(feats["High"].iloc[i])
            low  = float(feats["Low"].iloc[i])

            hit_tp = hit_sl = False
            if pos["side"] == "BUY":
                hit_tp = high >= pos["tp"]
                hit_sl = low <= pos["sl"]
            else:
                hit_tp = low <= pos["tp"]
                hit_sl = high >= pos["sl"]

            outcome = None
            rr = np.nan
            if hit_tp and not hit_sl:
                outcome = "tp"
                rr = abs((pos["tp"] - pos["entry"]) / (pos["entry"] - pos["sl"]))
            elif hit_sl and not hit_tp:
                outcome = "sl"
                rr = -1.0
            else:

                if (i - entry_idx) >= timeout_bars:
                    outcome = "timeout"
                    rr = 0.0

            if outcome is not None:
                trades.append({
                    "entry_idx": entry_idx,
                    "exit_idx": i,
                    "side": pos["side"],
                    "entry": pos["entry"],
                    "sl": pos["sl"],
                    "tp": pos["tp"],
                    "outcome": outcome,
                    "rr": rr,
                    "conf": pos["conf"],
                })
                pos = None
                entry_idx = None

    if len(trades) == 0:
        return pd.DataFrame(), pd.Series(dtype=float, name="equity"), pd.Series(dtype=float, name="equity_R")
    
    tr = pd.DataFrame(trades)
    r = tr["rr"].fillna(0.0).values
    equity = pd.Series(np.cumprod(1 + risk_frac * r), name="equity") 
    equity_R = pd.Series(np.r_[0.0, np.cumsum(r)], name="equity_R")   
    return tr, equity, equity_R


def metrics_from_trades(tr: pd.DataFrame, equity: pd.Series):
    if tr.empty or equity.empty:
        return {
            "num_trades": 0, "win_rate": np.nan, "expectancy_R": np.nan,
            "avg_win_R": np.nan, "avg_loss_R": np.nan, "max_dd": np.nan
        }

    num_trades = len(tr)
    win_rate = (tr["outcome"] == "tp").mean()

    wins = tr.loc[tr["rr"] > 0, "rr"]
    losses = tr.loc[tr["rr"] < 0, "rr"]
    avg_win = wins.mean() if not wins.empty else np.nan
    avg_loss = losses.mean() if not losses.empty else np.nan
    expectancy = tr["rr"].mean()
    ec = equity.copy()
    dd = (ec / ec.cummax() - 1).min()

    return {
        "num_trades": int(num_trades),
        "win_rate": float(win_rate),
        "expectancy_R": float(expectancy),
        "avg_win_R": float(avg_win) if pd.notna(avg_win) else np.nan,
        "avg_loss_R": float(avg_loss) if pd.notna(avg_loss) else np.nan,
        "max_dd": float(dd) if pd.notna(dd) else np.nan,
    }


def calibration_from_trades(tr: pd.DataFrame, bins=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0)):
    if tr.empty:
        return pd.DataFrame(columns=["bin_left", "bin_right", "n", "win_rate"])
    conf = tr["conf"].values
    outcomes = (tr["outcome"] == "tp").astype(int).values
    edges = np.array(bins)
    rows = []
    for i in range(len(edges)-1):
        lo, hi = edges[i], edges[i+1]
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            rows.append({"bin_left": lo, "bin_right": hi, "n": 0, "win_rate": np.nan})
        else:
            rows.append({"bin_left": lo, "bin_right": hi, "n": int(mask.sum()), "win_rate": float(outcomes[mask].mean())})
    return pd.DataFrame(rows)


def plot_and_save_equity(equity: pd.Series, outpath: str, title: str = "Equity Curve"):
    plt.figure()
    if not equity.empty:
        plt.plot(equity.values)
    plt.title(title)
    plt.xlabel("Trade #")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_and_save_hist(tr: pd.DataFrame, outpath: str):
    plt.figure()
    if not tr.empty:
        plt.hist(tr["rr"].dropna().values, bins=30)
    plt.title("Distribution of R multiples")
    plt.xlabel("R multiple")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_and_save_calibration(cal: pd.DataFrame, outpath: str):
    plt.figure()
    if not cal.empty:
        centers = (cal["bin_left"] + cal["bin_right"]) / 2
        plt.plot(centers, cal["win_rate"], marker="o")
        plt.plot([0.5, 1.0], [0.5, 1.0], linestyle="--")
        plt.ylim(0, 1)
    plt.title("Confidence Calibration")
    plt.xlabel("Confidence bin center")
    plt.ylabel("Win rate")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate the advisor bot on historical data.")
    parser.add_argument("--pair", default="EUR/USD", help='Pair name (e.g., "EUR/USD", "GBP/USD")')
    parser.add_argument("--interval", default="1h", choices=["1h", "4h", "1d"], help="Bar interval")
    parser.add_argument("--period", default="60d", help='History window (e.g., "60d", "90d", "1y")')
    parser.add_argument("--spread", type=float, default=1.0, help="Spread (pips)")
    parser.add_argument("--slip", type=float, default=0.5, help="Slippage (pips)")
    parser.add_argument("--timeout", type=int, default=200, help="Timeout in bars if neither SL/TP hits")
    parser.add_argument("--risk-frac", type=float, default=0.01, help="Fraction of equity risked per trade (for compounding equity)")
    parser.add_argument("--outdir", default="evaluation_output", help="Output directory")
    args = parser.parse_args()

    PAIRS = {
        "EUR/USD": "EURUSD=X", "USD/JPY": "USDJPY=X", "USD/SGD": "USDSGD=X",
        "GBP/USD": "GBPUSD=X", "AUD/USD": "AUDUSD=X", "EUR/GBP": "EURGBP=X",
        "EUR/JPY": "EURJPY=X", "GBP/JPY": "GBPJPY=X",
    }
    if args.pair not in PAIRS:
        raise SystemExit(f"Unknown pair: {args.pair}. Choices: {list(PAIRS.keys())}")

    os.makedirs(args.outdir, exist_ok=True)
    sym = PAIRS.args.pair if hasattr(PAIRS, 'args') else PAIRS[args.pair]

    df = yf.download(sym, interval=args.interval, period=args.period)
    if df is None or df.empty:
        raise SystemExit("No data returned from Yahoo Finance. Try a different period/interval.")

    # Features
    feat = df.copy()
    feat["SMA20"] = sma(feat["Close"], 20)
    feat["SMA50"] = sma(feat["Close"], 50)
    feat["RSI14"] = rsi(feat["Close"], 14)
    feat["ATR14"] = atr(feat, 14)
    feat = feat.dropna()

    trades, equity, equity_R = simulate_trades(
        feat, args.interval,
        spread_pips=args.spread,
        slip_pips=args.slip,
        timeout_bars=args.timeout,
        risk_frac=args.risk_frac
    )
    metrics = metrics_from_trades(trades, equity)
    cal = calibration_from_trades(trades)

    # Save artifacts
    trades.to_csv(os.path.join(args.outdir, "trades.csv"), index=False)
    pd.DataFrame([metrics]).to_csv(os.path.join(args.outdir, "summary.csv"), index=False)

    plot_and_save_equity(equity, os.path.join(args.outdir, "equity_curve.png"), title="Equity Curve (account-style, compounding)")
    plot_and_save_equity(equity_R, os.path.join(args.outdir, "equity_R_curve.png"), title="R-Equity (cumulative R)")
    plot_and_save_hist(trades, os.path.join(args.outdir, "hist_rr.png"))
    plot_and_save_calibration(cal, os.path.join(args.outdir, "calibration.png"))

    # Markdown summary
    md = []
    md.append(f"# Evaluation Summary — {args.pair} ({args.interval}, {args.period})")
    md.append("")
    md.append("**Assumptions:** entry on next bar open; simple spread + slippage costs; timeout if neither level is hit.")
    md.append("**Rules:** BUY if SMA20>SMA50 & RSI<70; SELL if SMA20<SMA50 & RSI>30; ATR-based SL/TP (1.5×ATR / 2.5×ATR).")
    md.append("")
    md.append("## Headline metrics")
    md.append(f"- Trades: {metrics['num_trades']}")
    md.append(f"- Win rate: {metrics['win_rate']:.1%}" if pd.notna(metrics['win_rate']) else "- Win rate: n/a")
    md.append(f"- Expectancy: {metrics['expectancy_R']:.2f} R" if pd.notna(metrics['expectancy_R']) else "- Expectancy: n/a")
    if pd.notna(metrics["avg_win_R"]): md.append(f"- Avg win: {metrics['avg_win_R']:.2f} R")
    if pd.notna(metrics["avg_loss_R"]): md.append(f"- Avg loss: {metrics['avg_loss_R']:.2f} R")
    md.append(f"- Max drawdown: {metrics['max_dd']:.2%}" if pd.notna(metrics['max_dd']) else "- Max drawdown: n/a")
    md.append("")
    md.append("## Figures (exported)")
    md.append("- equity_curve.png — strategy equity over trades (compounding with risk_frac).")
    md.append("- equity_R_curve.png — cumulative R (no compounding).")
    md.append("- hist_rr.png — distribution of realized R multiples.")
    md.append("- calibration.png — confidence vs. actual win rate.")
    md.append("")
    md.append("## Reproducibility")
    md.append("Tune parameters to test robustness (fees, timeout, other pairs, risk fraction).")

    with open(os.path.join(args.outdir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"Done. Outputs written to: {args.outdir}")
    print("Include summary.md and the PNGs in your report.")


if __name__ == "__main__":
    main()
