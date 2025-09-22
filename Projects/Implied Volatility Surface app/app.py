# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Tuple, List

import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata
import plotly.graph_objects as go

st.set_page_config(page_title="Implied Volatility Surface", layout="wide")

# =======================
# Helpers
# =======================
def bs_call_price(S, K, T, r_, sigma, q_=0.0):
    if T <= 0 or S <= 0 or K <= 0 or sigma <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r_ - q_ + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q_ * T) * norm.cdf(d1) - K * np.exp(-r_ * T) * norm.cdf(d2)

def implied_vol(mid_price, S, K, T, r_, q_=0.0):
    if T <= 0 or mid_price <= 0 or S <= 0 or K <= 0:
        return np.nan
    f = lambda sig: bs_call_price(S, K, T, r_, sig, q_) - mid_price
    try:
        return brentq(f, 1e-6, 5.0)
    except Exception:
        return np.nan

# =======================
# Yahoo data (cached)
# =======================
@st.cache_data(ttl=3600, show_spinner=False)
def get_spot_and_exps(ticker: str) -> Tuple[float, List[pd.Timestamp]]:
    tk = yf.Ticker(ticker)
    # Get spot from recent history (safer than info)
    hist = tk.history(period="5d", auto_adjust=False)
    if hist.empty:
        raise RuntimeError("Failed to fetch recent price history.")
    spot = float(hist["Close"].iloc[-1])

    # Parse option expiration dates
    expirations = tk.options or []
    exp_dates = [pd.Timestamp(e) for e in expirations]
    return spot, exp_dates

@st.cache_data(ttl=900, show_spinner=False)
def fetch_calls_for_expiration(ticker: str, exp_str: str) -> pd.DataFrame:
    """Return a SERIALIZABLE DataFrame of calls for a single expiration."""
    tk = yf.Ticker(ticker)
    oc = tk.option_chain(exp_str)
    calls = oc.calls.copy()
    # Keep only the columns we need and ensure primitive types
    keep = ["contractSymbol", "lastTradeDate", "strike", "bid", "ask", "lastPrice", "volume", "openInterest"]
    for col in keep:
        if col not in calls.columns:
            calls[col] = np.nan
    calls = calls[keep].copy()
    # Normalize dtypes
    calls["strike"] = calls["strike"].astype(float)
    for c in ["bid", "ask", "lastPrice"]:
        calls[c] = pd.to_numeric(calls[c], errors="coerce")
    return calls

def build_iv_table(
    ticker: str,
    r: float,
    q: float,
    y_axis: str,
    min_strike_pct: float,
    max_strike_pct: float,
    min_dte_days: int,
    max_expirations: int,
    iv_clip: Tuple[float, float],
) -> Tuple[pd.DataFrame, float]:
    today = pd.Timestamp("today").normalize()
    S, all_expirations = get_spot_and_exps(ticker)
    # Filter expirations by DTE and cap how many to pull (avoid rate limits / speed)
    exp_dates = [e for e in all_expirations if (e - today).days >= min_dte_days]
    exp_dates = sorted(exp_dates)[:max_expirations]
    if not exp_dates:
        raise RuntimeError(f"No expirations ≥ {min_dte_days} DTE found.")

    lo = S * (min_strike_pct / 100.0)
    hi = S * (max_strike_pct / 100.0)

    rows = []
    for exp in exp_dates:
        try:
            calls = fetch_calls_for_expiration(ticker, exp.strftime("%Y-%m-%d"))
        except Exception:
            continue

        # Basic hygiene
        calls = calls[(calls["bid"] > 0) & (calls["ask"] > 0)]
        calls = calls[(calls["strike"] >= lo) & (calls["strike"] <= hi)]
        if calls.empty:
            continue

        # Build rows
        for _, rr in calls.iterrows():
            mid = float((rr["bid"] + rr["ask"]) / 2.0) if np.isfinite(rr["bid"]) and np.isfinite(rr["ask"]) else np.nan
            rows.append(
                dict(
                    expirationDate=exp,
                    strike=float(rr["strike"]),
                    bid=float(rr["bid"]),
                    ask=float(rr["ask"]),
                    mid=mid,
                )
            )

    if not rows:
        raise RuntimeError("No option rows after filtering; widen strike window or lower min DTE.")

    df = pd.DataFrame(rows)
    df["daysToExpiration"] = (df["expirationDate"] - today).dt.days
    df = df[df["daysToExpiration"] >= min_dte_days].copy()
    df["timeToExpiration"] = df["daysToExpiration"] / 365.0
    df["moneyness"] = df["strike"] / S

    # Implied vols
    df["iv"] = df.apply(
        lambda rrow: implied_vol(
            mid_price=rrow["mid"], S=S, K=rrow["strike"],
            T=rrow["timeToExpiration"], r_=r, q_=q
        ),
        axis=1,
    )

    df = df.dropna(subset=["iv"]).copy()
    df["iv_pct"] = df["iv"] * 100.0
    df = df[(df["iv_pct"] >= iv_clip[0]) & (df["iv_pct"] <= iv_clip[1])]

    if df.empty:
        raise RuntimeError("All IV points filtered out (clip too tight or quotes inconsistent).")

    return df, S

def make_surface_figure(
    df: pd.DataFrame,
    y_axis: str,
    t_bins: int,
    k_bins: int,
    title_suffix: str,
):
    # Choose Y
    if y_axis.lower().startswith("mone"):
        Y = df["moneyness"].to_numpy()
        y_label = "Moneyness (K / S)"
    else:
        Y = df["strike"].to_numpy()
        y_label = "Strike ($)"

    X = df["timeToExpiration"].to_numpy()
    Z = df["iv_pct"].to_numpy()

    if len(df) < 5:
        raise RuntimeError("Too few valid IV points to build a surface.")

    # Regular grid
    ti = np.linspace(X.min(), X.max(), max(10, int(t_bins)))
    ki = np.linspace(Y.min(), Y.max(), max(10, int(k_bins)))
    Tg, Kg = np.meshgrid(ti, ki)
    Zi = griddata(points=(X, Y), values=Z, xi=(Tg, Kg), method="linear")
    Zi = np.ma.array(Zi, mask=np.isnan(Zi))

    fig = go.Figure(
        data=[
            go.Surface(
                x=Tg, y=Kg, z=Zi,
                colorscale="Viridis",
                colorbar_title="IV (%)",
                showscale=True,
            )
        ]
    )
    fig.add_scatter3d(
        x=X, y=Y, z=Z,
        mode="markers",
        marker=dict(size=3, opacity=0.6),
        name="Observed"
    )
    fig.update_layout(
        title=f"Implied Volatility Surface {title_suffix}",
        scene=dict(
            xaxis_title="Time to Expiration (years)",
            yaxis_title=y_label,
            zaxis_title="Implied Volatility (%)",
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        height=720,
    )
    return fig

# =======================
# UI
# =======================
st.title("📈 Implied Volatility Surface")

with st.sidebar:
    st.header("Parameters")
    ticker = st.text_input("Ticker", value="TSLA").strip().upper()
    col_rq = st.columns(2)
    with col_rq[0]:
        r = st.number_input("Risk-free r (annual, dec.)", value=0.015, step=0.001, format="%.4f")
    with col_rq[1]:
        q = st.number_input("Dividend yield q (annual, dec.)", value=0.013, step=0.001, format="%.4f")

    y_axis = st.selectbox("Y axis", options=["moneyness", "strike"], index=0)
    col_strk = st.columns(2)
    with col_strk[0]:
        min_strike_pct = st.number_input("Min strike (% of spot)", value=80.0, step=1.0)
    with col_strk[1]:
        max_strike_pct = st.number_input("Max strike (% of spot)", value=120.0, step=1.0)

    min_dte_days = st.number_input("Min DTE (days)", value=7, step=1, min_value=0)
    max_expirations = st.slider("Max expirations to fetch", min_value=1, max_value=24, value=12)

    col_bins = st.columns(2)
    with col_bins[0]:
        t_bins = st.slider("Time bins", min_value=10, max_value=120, value=60)
    with col_bins[1]:
        k_bins = st.slider("Strike/Moneyness bins", min_value=10, max_value=120, value=50)

    iv_clip_low, iv_clip_high = st.slider("Keep IV range (%)", 1.0, 300.0, (1.0, 200.0))
    show_table = st.checkbox("Show raw IV table", value=False)
    dl_btn = st.checkbox("Enable CSV download", value=True)

go_btn = st.button("Build Surface", type="primary")

# =======================
# Main action
# =======================
if go_btn:
    try:
        with st.spinner("Fetching options and computing IVs…"):
            df, S = build_iv_table(
                ticker=ticker,
                r=r,
                q=q,
                y_axis=y_axis,
                min_strike_pct=min_strike_pct,
                max_strike_pct=max_strike_pct,
                min_dte_days=min_dte_days,
                max_expirations=max_expirations,
                iv_clip=(iv_clip_low, iv_clip_high),
            )

        st.success(f"Spot price S ≈ {S:,.2f}. Using {df.shape[0]} IV points from {df['expirationDate'].nunique()} expirations.")

        # Surface
        fig = make_surface_figure(
            df=df,
            y_axis=y_axis,
            t_bins=t_bins,
            k_bins=k_bins,
            title_suffix=f"— {ticker} (calls, y={y_axis})",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Optional table
        if show_table:
            st.markdown("#### IV Points Used")
            show_cols = ["expirationDate","daysToExpiration","timeToExpiration","strike","moneyness","mid","iv","iv_pct"]
            st.dataframe(df[show_cols].sort_values(["expirationDate","strike"]).reset_index(drop=True), use_container_width=True, height=360)

        # Download
        if dl_btn:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name=f"{ticker}_iv_points.csv", mime="text/csv")

        # Small tips
        with st.expander("Tips & Troubleshooting"):
            st.write(
                "- If the surface looks sparse or has holes, reduce `Min DTE`, widen the strike window, or increase `Max expirations`.\n"
                "- If nothing shows up, relax the **IV range** filter or check that the ticker actually has listed options.\n"
                "- The scatter markers show raw observed IV points; the surface is a linear interpolation grid."
            )

    except Exception as e:
        st.error(f"⚠️ {type(e).__name__}: {e}")

else:
    st.info("Set your parameters in the left panel, then click **Build Surface**.")

# Footer
st.caption("Note: IVs are computed from call mid-prices using Black–Scholes with continuous dividend yield (q).")

