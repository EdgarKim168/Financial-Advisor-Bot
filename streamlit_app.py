# streamlit_app.py — Financial Advisor Bot
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from contextlib import contextmanager
import yfinance as yf

# ---------- Page config ----------
st.set_page_config(page_title="Financial Advisor Bot", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
div.stButton>button { border-radius: 9999px; padding: 0.55rem 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ---------- Tooltip CSS (for hover popouts) ----------
TOOLTIP_CSS = """
<style>
.tooltip { position:relative; display:inline-block; cursor:help; }
.tooltip .tiptext {
  visibility:hidden; width:260px; background:#111; color:#fff; text-align:left;
  border-radius:8px; padding:10px; position:absolute; z-index:1000; left:105%;
  top:50%; transform:translateY(-50%); box-shadow:0 6px 18px rgba(0,0,0,.3);
}
.tooltip:hover .tiptext { visibility:visible; }
.tooltip .k { font-weight:600; }
</style>
"""
st.markdown(TOOLTIP_CSS, unsafe_allow_html=True)

def render_abbrev_sidebar(abbr: dict, title: str = "Key terms"):
    st.markdown(f"**{title}**")
    st.caption("Hover the term below to see the meaning.")
    html = []
    for k, v in abbr.items():
        html.append(
            f'<div style="margin:4px 0">'
            f'  <span class="tooltip"><span class="k">{k}</span>'
            f'    <span class="tiptext">{v}</span>'
            f'  </span>'
            f'</div>'
        )
    st.markdown("\n".join(html), unsafe_allow_html=True)


# ---------- Helpers ----------
@contextmanager
def busy(msg="Working..."):
    with st.spinner(msg):
        yield

@st.cache_data(ttl=300, show_spinner=False)
def load_yahoo_fx(symbol="EURUSD=X", interval="1h", period="60d") -> pd.DataFrame:
    df = yf.download(symbol, interval=interval, period=period)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {symbol} ({interval}/{period})")
    if "Volume" not in df:
        df["Volume"] = 0.0
    return df[["Open", "High", "Low", "Close", "Volume"]]

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

def as_float(x):
    try:
        if isinstance(x, (pd.Series, np.ndarray, list, tuple)):
            x = np.asarray(x).ravel()[-1]
        return float(x)
    except Exception:
        return float("nan")

def fmt(v, p=5):
    return f"{v:.{p}f}" if np.isfinite(v) else "n/a"

# ---------- Strategy & backtest ----------
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
        sl, tp = price - risk_mult_sl*atr14, price + risk_mult_tp*atr14
        conf = min(0.95, 0.55 + (sma20 - sma50) / (0.5*atr14 + 1e-9))
    elif signal == "SELL":
        sl, tp = price + risk_mult_sl*atr14, price - risk_mult_tp*atr14
        conf = min(0.95, 0.55 + (sma50 - sma20) / (0.5*atr14 + 1e-9))
    else:
        sl = tp = np.nan
        conf = 0.45

    return dict(signal=signal, price=price, sl=sl, tp=tp,
                rsi=rsi14, atr=atr14, confidence=float(np.clip(conf, 0.05, 0.95)),
                rationale="; ".join(reason) if reason else "Mixed signals")

def backtest_sma(df_feat: pd.DataFrame):
    d = df_feat.copy()
    d["pos"] = np.where(d["SMA20"] > d["SMA50"], 1, np.where(d["SMA20"] < d["SMA50"], -1, 0))
    d["ret"] = d["Close"].pct_change().fillna(0)
    d["strat_ret"] = d["pos"].shift(1).fillna(0) * d["ret"]
    equity = (1 + d["strat_ret"]).cumprod()
    cagr = (equity.iloc[-1]) ** (365 * 24 / len(equity)) - 1 if len(equity) > 50 else np.nan
    dd = (equity / equity.cummax() - 1).min()
    hit = (d["strat_ret"] > 0).mean()
    return {"CAGR": float(cagr) if pd.notna(cagr) else None,
            "Max Drawdown": float(dd),
            "Hit Rate": float(hit)}, equity

# ---------- Text builders ----------
def expected_move_pct(rec, k: float = 2.0) -> float:
    if rec["price"] and np.isfinite(rec["atr"]):
        return float((k * rec["atr"]) / rec["price"] * 100)
    return 0.0

def horizon_from_interval(interval: str) -> str:
    return {"1h": "the next few days", "4h": "1–2 weeks", "1d": "1–3 months"}.get(interval, "the near term")

def format_recommendation_text(pair_name: str, interval: str, rec: dict) -> str:
    side = rec["signal"]
    conf = int(round(rec["confidence"] * 100))
    exp = expected_move_pct(rec, k=2.0)
    hz  = horizon_from_interval(interval)
    text = [
        f"**{side} {pair_name}** (confidence **{conf}%**).",
        f"Reasoning: {rec['rationale']} · RSI={rec['rsi']:.1f}, ATR={rec['atr']:.5f}.",
    ]
    if side in ("BUY", "SELL") and np.isfinite(rec["sl"]) and np.isfinite(rec["tp"]):
        dir_word = "upside" if side == "BUY" else "downside"
        flip = "SMA20 below SMA50" if side == "BUY" else "SMA20 above SMA50"
        text.append(
            f"Plan: entry ~ **{rec['price']:.5f}**, **SL** {rec['sl']:.5f}, **TP** {rec['tp']:.5f}. "
            f"Typical {dir_word} over {hz} ≈ **{exp:.1f}%** (ATR-based)."
        )
        text.append(f"Exit early if {flip} or RSI crosses 50 against the position.")
    else:
        text.append("Signals mixed; stay flat and wait for a cleaner setup.")
    return "\n\n".join(text)

def format_full_trade_plan_text(pair_name: str, interval: str, rec: dict) -> str:
    """Full plan: tidy layout + beginner explainer. No 'Invalidation' or 'Alternative plays'."""
    hz   = horizon_from_interval(interval)
    side = rec["signal"]
    conf = int(round(rec["confidence"] * 100))
    price = rec["price"]
    atr_v = rec["atr"]

    lines = [
        f"### Full Trade Plan — {pair_name}",
        f"**Bias:** {side} · **Confidence:** {conf}% · **Horizon:** {hz}",
        "",  # spacer before details
    ]

    if side in ("BUY", "SELL") and np.isfinite(rec["sl"]) and np.isfinite(rec["tp"]):
        dir_mult   = 1 if side == "BUY" else -1
        tp1        = price + dir_mult * 1.2 * atr_v
        tp2        = price + dir_mult * 2.0 * atr_v
        add_zone_a = price - dir_mult * 0.6 * atr_v
        add_zone_b = price + dir_mult * 0.3 * atr_v
        add_lo, add_hi = (min(add_zone_a, add_zone_b), max(add_zone_a, add_zone_b))
        rr_est     = abs((rec["tp"] - price) / (price - rec["sl"])) if price != rec["sl"] else np.nan

        lines += [
            "**Details:**",
            f"- **Context:** {rec['rationale']} (RSI={rec['rsi']:.1f}, ATR={atr_v:.5f})",
            f"- **Entry:** ~ **{price:.5f}** *(market or limit)*",
            f"- **Hard SL:** **{rec['sl']:.5f}**",
            f"- **Primary TP:** **{rec['tp']:.5f}**",
            f"- **Scale TP1 / TP2:** {tp1:.5f} / {tp2:.5f}",
            f"- **Scale zone (adds):** {add_lo:.5f} → {add_hi:.5f}",
            f"- **Position sizing:** risk 1% of equity to hard SL",
            f"- **Est. R:R:** {rr_est:.2f}",
            "",
            "---",
            "**Key for newcomers:**",
            "- **Entry** is where you aim to get filled. *Market* = fill now; *Limit* = only at your price (may miss).",
            "- **Hard SL** is your automatic exit to cap the loss if price moves against you.",
            "- **TP / TP1 / TP2** are profit targets; you can take partial profit at TP1 and let the rest run.",
            "- **Scale zone** is an optional area to add small position size if price wiggles before moving.",
            "- **Position sizing (1%)** means risk at most 1% of your account between entry and SL.",
            "- **R:R** is reward-to-risk (e.g., **1.6** means you aim to win 1.6 for every 1 you risk).",
        ]
    else:
        lines += [
            "**Details:**",
            f"- **Context:** {rec['rationale']} (RSI={rec['rsi']:.1f}, ATR={atr_v:.5f})",
            "- **No trade:** signals are mixed; wait for clearer alignment.",
            "",
            "---",
            "**Key for newcomers:**",
            "- **No trade** can be the best trade when signals disagree.",
            "- Wait for **SMA20 vs SMA50** to align and **RSI** to support the direction.",
        ]

    return "\n".join(lines)


# ---------- Sidebar (re-ordered with placeholders so Bars sits under Settings) ----------
st.title("Financial Advisor Bot")
PAIRS = {
    "EUR/USD": "EURUSD=X", "USD/JPY": "USDJPY=X", "USD/SGD": "USDSGD=X",
    "GBP/USD": "GBPUSD=X", "AUD/USD": "AUDUSD=X", "EUR/GBP": "EURGBP=X",
    "EUR/JPY": "EURJPY=X", "GBP/JPY": "GBPJPY=X",
}

with st.sidebar:
    st.markdown("### Settings")
    # create placeholder containers in the order we want to render
    settings_box = st.container()   # select boxes live here
    bars_box = st.container()       # Bars slider will be filled after data loads
    st.markdown("---")
    abbr_box = st.container()       # Abbreviations at the bottom

# fill settings first
with settings_box:
    pair_name = st.selectbox("Pair", list(PAIRS.keys()), index=0, key="pair_select")
    interval_map = {"1h": "1 hour", "4h": "4 hours", "1d": "1 day"}
    interval = st.radio(
        "Interval",
        options=list(interval_map.keys()),       # keep codes!
        index=0,                                 # 0="1h", set 2 if you want "1d" default
        horizontal=True,
        format_func=lambda v: interval_map[v],
        key="interval_radio",
    )

    # History window radio (non-typable) with label mapping
    period_map = {"30d": "30 days", "60d": "60 days", "90d": "90 days", "1y": "1 year"}
    period = st.radio(
        "History window",
        options=list(period_map.keys()),
        index=1,                                 # 1="60d"
        horizontal=True,
        format_func=lambda v: period_map[v],
        key="period_radio",
    )

symbol = PAIRS[pair_name]

# ---------- Load & features ----------
df = load_yahoo_fx(symbol=symbol, interval=interval, period=period)
feat_full = df.copy()
feat_full["SMA20"] = sma(feat_full["Close"], 20)
feat_full["SMA50"] = sma(feat_full["Close"], 50)
feat_full["RSI14"] = rsi(feat_full["Close"], 14)
feat_full["ATR14"] = atr(feat_full, 14)
feat_clean = feat_full.dropna()
latest = feat_clean.iloc[-1] if not feat_clean.empty else feat_full.iloc[-1]
rec = make_recommendation(latest)

# now fill the Bars slider in its reserved spot (right under Settings)
total_bars = int(len(df))
slider_key = f"bars|{symbol}|{interval}|{period}"
with bars_box:
    if total_bars < 3:
        n_bars = total_bars
        st.caption(f"{total_bars} bars loaded: {df.index.min().date()} → {df.index.max().date()} "
                   "(slider disabled: not enough data)")
    else:
        min_slider = max(2, min(20, total_bars - 1))
        max_slider = total_bars
        step_size  = max(1, total_bars // 20)
        n_bars = st.slider("Bars", min_value=min_slider, max_value=max_slider,
                           value=max_slider, step=step_size, key=slider_key)
        st.caption(f"{total_bars} bars loaded: {df.index.min().date()} → {df.index.max().date()}")

# finally show Abbreviations at the very bottom of the sidebar
with abbr_box:
    ABBR = {
        "SMA": "Simple Moving Average — rolling average of price over N periods.",
        "RSI": "Relative Strength Index — momentum oscillator (0–100).",
        "ATR": "Average True Range — average volatility of recent candles.",
        "SL":  "Stop Loss — price to cut the loss.",
        "TP":  "Take Profit — price to take profit.",
        "R:R": "Risk-to-Reward ratio — potential loss vs. potential gain.",
    }
    render_abbrev_sidebar(ABBR, title="Key terms")

# ---------- Quick Questions ----------
if "action" not in st.session_state:
    st.session_state.action = None

st.subheader("Quick Questions")
questions = [
    {"label": "What’s the latest recommendation?", "key": "latest"},
    {"label": "Why that advice?",                  "key": "why"},
    {"label": "What SL/TP do you suggest?",        "key": "sltp"},
    {"label": "Run a quick backtest",              "key": "backtest"},
    {"label": "Give me a full trade plan",         "key": "tradeplan"},
]
row1 = questions[:3]
row2 = questions[3:]
for row in (row1, row2):
    cols = st.columns(len(row), gap="small")
    for col, q in zip(cols, row):
        with col:
            if st.button(q["label"], key=f"btn_{q['key']}", use_container_width=True):
                st.session_state.action = q["key"]

def chat_user(x):
    with st.chat_message("user"):
        st.markdown(x)

def chat_bot(x):
    with st.chat_message("assistant"):
        st.markdown(x)

act = st.session_state.action
if act == "latest":
    chat_user("What’s the latest recommendation?")
    chat_bot(format_recommendation_text(pair_name, interval, rec))

elif act == "why":
    chat_user("Why that advice?")
    sma20_v = as_float(latest.get("SMA20"))
    sma50_v = as_float(latest.get("SMA50"))
    rsi_v   = as_float(rec.get("rsi"))
    chat_bot(
        f"Rationale: {rec['rationale']}  \n"
        f"SMA20={fmt(sma20_v)}, "
        f"SMA50={fmt(sma50_v)}, "
        f"RSI={fmt(rsi_v, 1)}."
    )

elif act == "sltp":
    chat_user("What SL/TP do you suggest?")
    if rec["signal"] in ("BUY", "SELL") and np.isfinite(rec["sl"]) and np.isfinite(rec["tp"]):
        chat_bot(f"Proposed **SL**: {rec['sl']:.5f} · **TP**: {rec['tp']:.5f} (ATR-based).")
    else:
        chat_bot("No SL/TP for **HOLD** or when signals are mixed.")

elif act == "backtest":
    chat_user("Run a quick backtest")
    with busy("Backtesting SMA20/50…"):
        if feat_clean.empty:
            chat_bot("Not enough data after indicators warm-up to run the backtest.")
        else:
            metrics, equity = backtest_sma(feat_clean)
            lines = [
                "**Backtest (SMA20/50, long/short flips)**",
                f"- CAGR: {metrics['CAGR']:.2%}" if metrics["CAGR"] is not None else "- CAGR: n/a",
                f"- Max Drawdown: {metrics['Max Drawdown']:.2%}",
                f"- Hit Rate: {metrics['Hit Rate']:.1%}",
            ]
            chat_bot("\n".join(lines))
            with st.chat_message("assistant"):
                st.line_chart(equity.rename("Equity Curve"))

elif act == "tradeplan":
    chat_user("Give me a full trade plan")
    chat_bot(format_full_trade_plan_text(pair_name, interval, rec))

st.session_state.action = None

# ---------- Market Chart ----------
st.subheader("Market Chart")
window = feat_full.iloc[-n_bars:].copy()
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=window.index,
    open=window["Open"], high=window["High"],
    low=window["Low"], close=window["Close"], name="Price"
))
fig.add_trace(go.Scatter(x=window.index, y=window["SMA20"], name="SMA20"))
fig.add_trace(go.Scatter(x=window.index, y=window["SMA50"], name="SMA50"))
fig.update_layout(height=540, xaxis_rangeslider_visible=False,
                  margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption(f"Showing last **{len(window)}** bars: {window.index.min().date()} → {window.index.max().date()}")
