"""Small smoke tests for configuration, feature transforms, and history helpers."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from arithmia_comp.config import PipelineConfig
from arithmia_comp.features import engineer_features
from search_history import compact_history_for_llm, record_trial, resume_parameters


class PipelineSmokeTests(unittest.TestCase):
    def test_feature_transform_has_expected_indicators(self):
        frame = pd.DataFrame({
            "VentricularRate": [70, 165], "AtrialRate": [70, 165],
            "QRSDuration": [90, 100], "PatientAge": [30, 55],
            "QOnset": [10, 10], "QOffset": [100, 110],
            "TOffset": [180, 190], "TAxis": [20, 40], "RAxis": [25, 45],
            "QTInterval": [400, 420], "QRSCount": [12, 27],
        })
        result = engineer_features(frame)
        self.assertIn("ind_class3_avnrt", result)
        self.assertIn("ind_class4_avrt", result)
        self.assertIn("ind_class6_saawr", result)
        self.assertNotIn("TAxis", result)

    def test_history_resume_and_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.json"
            record_trial(path, params={"hidden_dim": 32}, score=0.4,
                         fold_metrics=[{"fold": 1, "macro_f1": 0.4}])
            history = __import__("search_history").load_search_history(path)
            params, score = resume_parameters(history, {"hidden_dim": 40, "n_splits": 5})
            self.assertEqual(params, {"hidden_dim": 32, "n_splits": 5})
            self.assertEqual(score, 0.4)
            self.assertIn("folds", compact_history_for_llm(history))

    def test_config_ignores_unknown_keys(self):
        config = PipelineConfig.from_dict({"n_splits": 3, "future_option": True})
        self.assertEqual(config.n_splits, 3)
        self.assertFalse(hasattr(config, "future_option"))


if __name__ == "__main__":
    unittest.main()
