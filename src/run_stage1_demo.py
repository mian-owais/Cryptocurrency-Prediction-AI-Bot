"""
run_stage1_demo.py
------------------
Small runner to execute Stage 1 pipeline end-to-end in a lightweight/demo mode.

It will:
 - fetch recent BTC/ETH data (few days)
 - compute indicators and save processed CSV
 - run the training pipeline for BTC (lightweight)

This script is defensive: it catches and prints errors like missing packages or network issues.
"""
import traceback
import sys

from src.data_pipeline import prepare_and_save
from src.model_train import train_models_for_coin


def main():
    try:
        print("[demo] Preparing processed data (2 days)...")
        df = prepare_and_save(["bitcoin", "ethereum"], days=2, out_path="data/processed/crypto_data.csv")
        print("[demo] Processed data rows:", len(df))
    except Exception as e:
        print("[demo] Data preparation failed:", e)
        traceback.print_exc()

    try:
        print("[demo] Training models for bitcoin (light)...")
        res = train_models_for_coin("bitcoin", processed_path="data/processed/crypto_data.csv", out_model_path="models/best_model_demo.pkl")
        print("[demo] Training result:", res)
    except Exception as e:
        print("[demo] Training failed:", e)
        traceback.print_exc()


if __name__ == "__main__":
    main()
