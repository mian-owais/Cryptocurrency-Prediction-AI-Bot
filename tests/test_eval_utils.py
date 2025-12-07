"""
test_eval_utils.py
-----------------
Unit tests for evaluation utilities.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.eval_utils import (
    compute_classification_metrics,
    get_worst_predictions
)


class TestEvalUtils(unittest.TestCase):
    def setUp(self):
        # Create sample prediction data
        self.sample_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2025-01-01', periods=100),
            'predicted_label': ['Increase', 'Decrease'] * 50,
            'actual_label': ['Increase'] * 75 + ['Decrease'] * 25,
            'predicted_prob': np.random.uniform(0.6, 1.0, 100),
            'abs_error': np.random.uniform(0, 0.2, 100)
        })

    def test_classification_metrics(self):
        """Test classification metrics computation."""
        metrics = compute_classification_metrics(self.sample_data)

        self.assertIsInstance(metrics, dict)
        self.assertTrue(0 <= metrics['accuracy'] <= 1)
        self.assertTrue(0 <= metrics['weighted_precision'] <= 1)
        self.assertTrue(0 <= metrics['weighted_recall'] <= 1)
        self.assertTrue(0 <= metrics['weighted_f1'] <= 1)

    def test_worst_predictions(self):
        """Test worst predictions identification."""
        # Test confidence-based worst predictions
        n_worst = 5
        worst_conf = get_worst_predictions(
            self.sample_data,
            n=n_worst,
            by='confidence'
        )

        self.assertEqual(len(worst_conf), n_worst)
        self.assertTrue(
            all(worst_conf['predicted_label'] != worst_conf['actual_label'])
        )

        # Test error-based worst predictions
        worst_error = get_worst_predictions(
            self.sample_data,
            n=n_worst,
            by='error'
        )

        self.assertEqual(len(worst_error), n_worst)
        self.assertTrue(
            worst_error['abs_error'].is_monotonic_decreasing
        )


if __name__ == '__main__':
    unittest.main()
