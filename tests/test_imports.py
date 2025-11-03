"""
Simple unit tests to ensure core modules import without syntax errors.

These tests are lightweight and do not require network or heavy training.
Run with: python -m pytest -q tests/test_imports.py
"""
def test_import_core_modules():
    import importlib

    modules = [
        'src.data_pipeline',
        'src.model_train',
        'src.model_predict',
        'src.strategy_engine',
        'src.explainability',
        'src.app.streamlit_app',
        'src.rl_agent',
    ]
    for m in modules:
        importlib.import_module(m)
