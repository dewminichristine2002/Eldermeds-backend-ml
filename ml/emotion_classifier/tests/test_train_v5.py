import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_v5_answer_only import LABELS, model_metrics, near_duplicate_groups, split_indices


class V5TrainingUtilityTests(unittest.TestCase):
    def test_exact_duplicate_texts_share_a_group(self):
        groups, audit = near_duplicate_groups(
            ["I feel very sad today.", "I feel very sad today.", "The bus is late."],
            threshold=0.99,
        )
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[0], groups[2])
        self.assertEqual(audit["exact_pairs"], 1)

    def test_group_splits_are_reproducible_and_disjoint(self):
        labels = np.asarray([label for label in LABELS for _ in range(20)])
        groups = np.arange(len(labels))
        first = split_indices(labels, groups)
        second = split_indices(labels, groups)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(sum(map(len, first)), len(labels))
        self.assertFalse(set(first[0]) & set(first[1]))
        self.assertFalse(set(first[0]) & set(first[2]))
        self.assertFalse(set(first[1]) & set(first[2]))

    def test_metrics_keep_seven_class_macro_f1(self):
        result = model_metrics(LABELS, LABELS)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["macro_f1"], 1.0)
        self.assertEqual(list(result["per_class"]), LABELS)


if __name__ == "__main__":
    unittest.main()
