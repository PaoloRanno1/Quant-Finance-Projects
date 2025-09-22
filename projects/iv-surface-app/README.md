# Implied Volatility Surface — Streamlit

An interactive Streamlit app that fetches option chains from Yahoo Finance and builds a **3D implied volatility (IV) surface** using Black–Scholes (calls, continuous dividend yield `q`). The plot overlays the raw observed IV points for transparency.
### Link to the app:https://quant-finance-projects-cwqgeqjiioi9w255pzghsc.streamlit.app/

## Features
- Live options data via `yfinance`
- IV from **call mid-prices** (`(bid+ask)/2`) with Brent root-finding
- **Moneyness (K/S)** or **Strike ($)** as Y-axis
- Interactive **Plotly 3D surface** + scatter of observed points
- Controls for strike window, min DTE, grid bins, IV clipping, expirations cap
- Caching and CSV download of computed IV table

## Quick Start

### Install
```bash
# (optional) create & activate a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt

