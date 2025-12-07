"""
explainability.py
-----------------
Purpose:
 - Compute SHAP values for model explainability
 - Produce summary plots and local explanation visuals

Key functions:
 - compute_shap_values(model, X): return shap values and expected value
 - plot_shap_summary(shap_values, X): create summary plot (returns matplotlib/plotly figure)
 - plot_shap_force(shap_values, X_row): create force/local explanation

Notes:
 - For tree-based models shap.TreeExplainer is recommended, for other models use KernelExplainer
 - This module keeps plotting functions separate from the Streamlit code so they can be tested independently
"""

from typing import Any, Tuple
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path


def compute_shap_values(model: Any, X: pd.DataFrame):
    """Compute SHAP values for the given model and dataset.

    Returns (explainer, shap_values)
    """
    try:
        # Prefer a TreeExplainer for tree models
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.Explainer(model, X)

    shap_values = explainer(X)
    return explainer, shap_values


def plot_shap_summary(shap_values, X: pd.DataFrame):
    """Generate a SHAP summary plot and return the matplotlib Figure.
    Streamlit can display matplotlib figures directly.
    """
    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    return fig


def save_shap_summary(shap_values, X: pd.DataFrame, out_path: str = "data/plots/shap_summary.png") -> str:
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    fig = plot_shap_summary(shap_values, X)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_shap_force(shap_values, X_row):
    """Generate a force plot (local explanation). Returns the plot object or HTML.
    Streamlit supports HTML/Plotly outputs; this function may need adaptation.
    """
    # For simple usage, return shap.force_plot as html (if available)
    try:
        f = shap.force_plot(shap_values.base_values, shap_values.values, X_row)
        return f
    except Exception:
        return None


def save_shap_force(shap_values, X_row, out_path: str = "data/plots/shap_force.html") -> str:
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    try:
        # shap.force_plot can be rendered to HTML
        html = shap.plots._force._save_html(
            shap_values.base_values, shap_values.values, X_row)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        return out_path
    except Exception:
        return ""


if __name__ == "__main__":
    print("explainability module — requires trained model and dataset to run")
