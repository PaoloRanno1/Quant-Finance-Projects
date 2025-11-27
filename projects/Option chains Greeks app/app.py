import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm


# ==========================================
# 1. Math & Logic (Black-Scholes Model)
# ==========================================

def black_scholes(S, K, T, r, sigma, option_type='Call'):
    """
    Calculates Price and Greeks for a single option.
    """
    # Safety against division by zero for T
    if T <= 1e-5:
        T = 1e-5

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Probabilities
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    N_neg_d1 = norm.cdf(-d1)
    N_neg_d2 = norm.cdf(-d2)

    # PDF for Greeks
    pdf_d1 = norm.pdf(d1)

    if option_type == 'Call':
        price = S * N_d1 - K * np.exp(-r * T) * N_d2
        delta = N_d1
        theta = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * N_d2)
        rho = K * T * np.exp(-r * T) * N_d2
        payoff = np.maximum(S - K, 0)

    else:  # Put
        price = K * np.exp(-r * T) * N_neg_d2 - S * N_neg_d1
        delta = N_d1 - 1
        theta = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * N_neg_d2)
        rho = -K * T * np.exp(-r * T) * N_neg_d2
        payoff = np.maximum(K - S, 0)

    # Gamma and Vega are the same for Calls and Puts
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * pdf_d1 * np.sqrt(T)

    return {
        'Price': price,
        'Payoff': payoff,
        'Delta': delta,
        'Gamma': gamma,
        'Theta': theta,
        'Vega': vega / 100,  # Per 1% change
        'Rho': rho / 100  # Per 1% change
    }


def calculate_portfolio_metrics(positions, spot_range, T, r, sigma):
    """
    Aggregates metrics for a list of positions over a range of spot prices.
    """
    aggregated = {
        'Price': np.zeros_like(spot_range),
        'Payoff': np.zeros_like(spot_range),
        'Delta': np.zeros_like(spot_range),
        'Gamma': np.zeros_like(spot_range),
        'Theta': np.zeros_like(spot_range),
        'Vega': np.zeros_like(spot_range),
        'Rho': np.zeros_like(spot_range)
    }

    for pos in positions:
        metrics = black_scholes(
            S=spot_range,
            K=pos['strike'],
            T=T,
            r=r,
            sigma=sigma,
            option_type=pos['type']
        )

        qty = pos['quantity']
        for key in aggregated:
            aggregated[key] += metrics[key] * qty

    return aggregated


# ==========================================
# 2. Streamlit UI Layout
# ==========================================

st.set_page_config(page_title="Options Visualizer", layout="wide")

st.title("📈 Options Portfolio Visualizer")
st.markdown("""
Visualize the Greeks and P&L for single options or complex strategies (Straddles, Iron Condors, etc.).
""")

# --- Sidebar: Controls ---
with st.sidebar:
    st.header("1. Market Parameters")

    # Market inputs
    current_spot = st.number_input("Spot Price ($)", value=100.0, step=1.0)
    volatility = st.slider("Volatility (σ)", 0.01, 1.50, 0.20, 0.01)
    time_to_expiry = st.slider("Time to Expiry (Years)", 0.01, 2.0, 0.5, 0.01)
    risk_free_rate = st.slider("Risk-Free Rate (r)", 0.0, 0.20, 0.05, 0.005)

    st.markdown("---")
    st.header("2. Portfolio Builder")

    # Initialize session state for positions
    if 'positions' not in st.session_state:
        st.session_state.positions = [
            {'id': 0, 'type': 'Call', 'strike': 100.0, 'quantity': 1}  # Default
        ]
        st.session_state.next_id = 1

    # Add Position Form
    with st.form("add_position"):
        c1, c2 = st.columns(2)
        with c1:
            opt_type = st.selectbox("Type", ["Call", "Put"])
            side = st.selectbox("Side", ["Long", "Short"])
        with c2:
            strike = st.number_input("Strike", value=current_spot)
            qty = st.number_input("Qty", value=1, min_value=1)

        if st.form_submit_button("Add Position"):
            final_qty = qty if side == "Long" else -qty
            st.session_state.positions.append({
                'id': st.session_state.next_id,
                'type': opt_type,
                'strike': strike,
                'quantity': final_qty
            })
            st.session_state.next_id += 1
            st.rerun()

    # Display Positions
    st.subheader("Current Positions")
    if not st.session_state.positions:
        st.warning("No positions.")
    else:
        for i, pos in enumerate(st.session_state.positions):
            p_side = "Long" if pos['quantity'] > 0 else "Short"
            p_qty = abs(pos['quantity'])
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"**{p_side} {p_qty}x {pos['type']}** @ {pos['strike']}")
            if col2.button("🗑️", key=f"del_{pos['id']}"):
                st.session_state.positions.pop(i)
                st.rerun()

# ==========================================
# 3. Visualization
# ==========================================

if st.session_state.positions:
    # 1. Determine Spot Range for X-Axis
    strikes = [p['strike'] for p in st.session_state.positions]
    avg_strike = np.mean(strikes) if strikes else current_spot

    # Create a range ±50% around the average strike or current spot
    x_min = min(current_spot * 0.7, avg_strike * 0.5)
    x_max = max(current_spot * 1.3, avg_strike * 1.5)
    spot_range = np.linspace(x_min, x_max, 200)

    # 2. Calculate Portfolio Metrics
    data = calculate_portfolio_metrics(
        st.session_state.positions,
        spot_range,
        time_to_expiry,
        risk_free_rate,
        volatility
    )

    # Calculate single point for current spot marker
    current_data = calculate_portfolio_metrics(
        st.session_state.positions,
        np.array([current_spot]),
        time_to_expiry,
        risk_free_rate,
        volatility
    )


    # 3. Plotting Helper
    def create_chart(metric_name, y_data, color, is_price=False):
        fig = go.Figure()

        # Main Curve (Value today)
        fig.add_trace(go.Scatter(
            x=spot_range, y=y_data,
            mode='lines', name=f'{metric_name} (Today)',
            line=dict(color=color, width=3)
        ))

        # Payoff Curve (Only for Price tab)
        if is_price:
            fig.add_trace(go.Scatter(
                x=spot_range, y=data['Payoff'],
                mode='lines', name='Payoff @ Expiry',
                line=dict(color='gray', width=2, dash='dash')
            ))

        # Current Spot Marker
        curr_val = current_data[metric_name][0]
        fig.add_trace(go.Scatter(
            x=[current_spot], y=[curr_val],
            mode='markers', name=f'Current Spot ({current_spot})',
            marker=dict(size=12, color='white', line=dict(width=2, color='black'))
        ))

        # Zero Line
        if not is_price:
            fig.add_hline(y=0, line_width=1, line_color="white", opacity=0.3)

        fig.update_layout(
            title=f"Portfolio {metric_name}",
            xaxis_title="Spot Price",
            yaxis_title=metric_name,
            template="plotly_dark",
            hovermode="x unified",
            height=500
        )
        return fig, curr_val


    # 4. Tabs
    tabs = st.tabs(["Price & Payoff", "Delta", "Gamma", "Theta", "Vega", "Rho"])

    with tabs[0]:
        fig, val = create_chart("Price", data['Price'], "#00CC96", is_price=True)
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Portfolio Value", f"${val:,.2f}")

    with tabs[1]:
        fig, val = create_chart("Delta", data['Delta'], "#EF553B")
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Net Delta", f"{val:.4f}")
        st.info("Delta: Rate of change of option price with respect to the underlying price.")

    with tabs[2]:
        fig, val = create_chart("Gamma", data['Gamma'], "#AB63FA")
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Net Gamma", f"{val:.4f}")
        st.info("Gamma: Rate of change of Delta.")

    with tabs[3]:
        fig, val = create_chart("Theta", data['Theta'], "#FFA15A")
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Net Theta", f"{val:.4f}")
        st.info("Theta: Time decay per day.")

    with tabs[4]:
        fig, val = create_chart("Vega", data['Vega'], "#19D3F3")
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Net Vega", f"{val:.4f}")
        st.info("Vega: Sensitivity to 1% change in volatility.")

    with tabs[5]:
        fig, val = create_chart("Rho", data['Rho'], "#FF6692")
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Net Rho", f"{val:.4f}")
        st.info("Rho: Sensitivity to 1% change in interest rates.")

else:
    st.info("👈 Add a position in the sidebar to get started!")
