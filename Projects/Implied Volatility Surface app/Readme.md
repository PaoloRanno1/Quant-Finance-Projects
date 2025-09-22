# Implied Volatility Surface (Streamlit)

Interactive Streamlit app that fetches option chains from Yahoo Finance and builds a 3D implied volatility (IV) surface using Black–Scholes (call mid-prices) with a continuous dividend yield. Includes robust filtering, interpolation, and overlay of observed IV points for transparency.
---
## App Controls

Ticker: e.g. TSLA, SPY, AAPL

r: Risk-free rate (annualized, decimal), e.g. 0.015

q: Dividend yield (annualized, decimal), e.g. 0.013

Y axis: moneyness (K/S) or strike ($)

Strike window: min/max % of spot (e.g. 80–120%)

Min DTE: minimum days to expiration

Max expirations: cap to fetch for speed/stability

Bins: interpolation grid density (time & strike/moneyness)

IV clip: keep IV within [%low, %high]

Show raw table: display computed IV points

Download CSV: export table

---
## How It Works (brief)

Fetch spot & option expirations with yfinance.

Filter calls by positive quotes, strike band, and min DTE.

Compute call IV for each row using Brent’s root finder on the Black–Scholes formula with continuous q.

Clean (drop NaNs, clip IV % range).

Interpolate (linear) to a regular grid for a smooth surface.

Render a Plotly 3D surface + scatter of original points.

Note: The surface shows an interpolated grid; the scatter points show where real quotes support the surface.
