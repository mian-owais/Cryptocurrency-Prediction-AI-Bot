from setuptools import setup, find_packages

setup(
    name="ai_trading_bot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "pandas",
        "numpy",
        "plotly",
        "scikit-learn",
        "requests",
        "reportlab",
        "streamlit-autorefresh"
    ],
)
