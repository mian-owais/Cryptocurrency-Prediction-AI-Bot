"""
streamlit_app.py
----------------
Streamlit dashboard for the AI Cryptocurrency Trading Bot.

Responsibilities:
 - Show live price (via CoinGecko)
 - Show model prediction for next interval
 - Show recommended action (BUY/SELL/HOLD)
 - Display SHAP explainability visualizations

This file wires together modules in src/ and provides a friendly UI. Start here when
building the front-end. The app uses placeholder flows if models/data are not yet available.
"""

import streamlit as st
import pandas as pd
import numpy as np
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

from src.data_pipeline import fetch_data, add_indicators, prepare_and_save
from src.model_predict import load_model, predict_next_from_data, load_processed_for_coin
from src.strategy_engine import generate_trade_signal, csp_portfolio_optimization
from src.explainability import compute_shap_values, plot_shap_summary, save_shap_summary

MODEL_PATH = "models/best_model.pkl"
PROCESSED_PATH = "data/processed/crypto_data.csv"


st.set_page_config(page_title="AI Crypto Trading Bot", layout="wide")


@st.cache_data(ttl=60)
def load_recent_data(coin: str = "bitcoin", days: int = 7) -> pd.DataFrame:
    df = fetch_data([coin], days=days)
    df = df.loc[coin]
    df = add_indicators(df)
    return df


def load_processed_coin(coin: str = "bitcoin") -> pd.DataFrame:
    try:
        return load_processed_for_coin(PROCESSED_PATH, coin=coin)
    except Exception:
        # fallback to fetching recent data
        return load_recent_data(coin, days=30)


def main():
    st.title("AI Cryptocurrency Trading Bot Dashboard")

    # auto-refresh every 30 seconds
    st_autorefresh(interval=30 * 1000, key="datarefresh")

    coin = st.selectbox("Select cryptocurrency", [
                        "bitcoin", "ethereum"], index=0)

    tabs = st.tabs(["📊 Market Data", "🤖 Prediction",
                   "💡 Recommendation", "🔍 Explainability"])

    # Market Data tab
    with tabs[0]:
        st.header("Market Data — Last 7 days")
        df_live = load_recent_data(coin, days=7)
        if df_live.empty:
            st.warning("No live data available")
        else:
            st.line_chart(df_live["close"].rename(
                "Price") if "close" in df_live.columns else df_live["price"].rename("Price"))
            st.dataframe(df_live.tail())

    # Prediction tab
    with tabs[1]:
        st.header("Model Prediction")
        model = None
        try:
            model = load_model(MODEL_PATH)
        except Exception:
            st.info(
                "No trained model found. Train a model and save to models/best_model.pkl")

        processed = None
        try:
            processed = load_processed_for_coin(PROCESSED_PATH, coin=coin)
        except Exception:
            processed = None

        if model is not None and processed is not None:
            try:
                result = predict_next_from_data(model, processed)
                st.metric("Last Price", f"${result['last_price']:.2f}")
                st.metric("Predicted Next",
                          f"${result['predicted_price']:.2f}")
                st.write(
                    f"Predicted change: {result['predicted_pct_change'] * 100:.2f}%")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.info("Model or processed data missing — running mock prediction")
            last = df_live.iloc[-1]
            last_price = float(last.get("close", last.get("price")))
            predicted = last_price * (1 + float(np.random.normal(0, 0.01)))
            st.metric("Last Price", f"${last_price:.2f}")
            st.metric("Predicted Next", f"${predicted:.2f}")

    # Recommendation tab
    with tabs[2]:
        st.header("Recommendation")
        try:
            if model is not None and processed is not None:
                res = predict_next_from_data(model, processed)
                action = generate_trade_signal(
                    res["last_price"], res["predicted_price"])
                color = "green" if action == "BUY" else (
                    "red" if action == "SELL" else "yellow")
                st.markdown(
                    f"### Action: <span style='color:{color}'>{action}</span>", unsafe_allow_html=True)
                st.write(res)
            else:
                st.info("No model available — showing mock recommendation")
                last = df_live.iloc[-1]
                last_price = float(last.get("close", last.get("price")))
                predicted = last_price * (1 + float(np.random.normal(0, 0.01)))
                action = generate_trade_signal(last_price, predicted)
                color = "green" if action == "BUY" else (
                    "red" if action == "SELL" else "yellow")
                st.markdown(
                    f"### Action: <span style='color:{color}'>{action}</span>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Recommendation error: {e}")

        # Simple portfolio optimizer demo
        st.subheader("Portfolio optimizer (discrete demo)")
        assets = [coin]
        expected_returns = np.array([0.01])
        alloc = csp_portfolio_optimization(assets, expected_returns)
        st.write("Suggested allocations:", alloc)

    # Explainability tab
    with tabs[3]:
        st.header("Explainability (SHAP)")
        if model is None or processed is None:
            st.info(
                "Need a trained model and processed dataset to show SHAP explanations")
        else:
            try:
                # Prepare X
                X = processed.select_dtypes(include=["number"]).copy()
                explainer, shap_vals = compute_shap_values(
                    model, X.iloc[-100:])
                fig = plot_shap_summary(shap_vals, X.iloc[-100:])
                st.pyplot(fig)
                # Save figure
                out = save_shap_summary(shap_vals, X.iloc[-100:])
                st.success(f"SHAP summary saved to {out}")
            except Exception as e:
                st.error(f"SHAP failed: {e}")


if __name__ == "__main__":
    main()
