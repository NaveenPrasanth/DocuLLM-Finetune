"""Tests for evaluation metrics."""

import json

import pytest

from src.evaluation.metrics import (
    compute_all_metrics,
    compute_anls,
    compute_exact_match,
    compute_field_f1,
    compute_json_validity,
    compute_per_field_metrics,
    compute_schema_compliance,
)


class TestFieldF1:
    def test_perfect_match(self, sample_flat_ground_truth):
        result = compute_field_f1(sample_flat_ground_truth, sample_flat_ground_truth)
        assert result["micro"]["f1"] == 1.0
        assert result["micro"]["precision"] == 1.0
        assert result["micro"]["recall"] == 1.0

    def test_partial_match(self, sample_flat_prediction, sample_flat_ground_truth):
        result = compute_field_f1(sample_flat_prediction, sample_flat_ground_truth)
        # Prediction has 10 fields, ground truth has 13
        # All 10 predicted fields are correct
        assert result["micro"]["precision"] == 1.0
        assert result["micro"]["recall"] == pytest.approx(10 / 13, abs=0.01)
        assert result["micro"]["f1"] > 0.0

    def test_no_match(self):
        pred = {"field_a": "value_a"}
        gt = {"field_b": "value_b"}
        result = compute_field_f1(pred, gt)
        assert result["micro"]["f1"] == 0.0

    def test_empty_prediction(self, sample_flat_ground_truth):
        result = compute_field_f1({}, sample_flat_ground_truth)
        assert result["micro"]["recall"] == 0.0

    def test_empty_ground_truth(self, sample_flat_prediction):
        result = compute_field_f1(sample_flat_prediction, {})
        assert result["micro"]["precision"] == 0.0


class TestExactMatch:
    def test_identical_json(self):
        j = '{"a": 1, "b": 2}'
        assert compute_exact_match(j, j) == 1.0

    def test_different_json(self):
        assert compute_exact_match('{"a": 1}', '{"a": 2}') == 0.0

    def test_reordered_keys(self):
        """JSON with same content but different key order should match."""
        j1 = '{"a": 1, "b": 2}'
        j2 = '{"b": 2, "a": 1}'
        assert compute_exact_match(j1, j2) == 1.0

    def test_invalid_json(self):
        assert compute_exact_match("not json", '{"a": 1}') == 0.0


class TestANLS:
    def test_identical_strings(self):
        assert compute_anls("hello world", "hello world") == 1.0

    def test_similar_strings(self):
        score = compute_anls("hello world", "hello worl")
        assert 0.5 < score < 1.0

    def test_completely_different(self):
        score = compute_anls("abc", "xyz")
        assert score == 0.0  # Below threshold

    def test_empty_strings(self):
        assert compute_anls("", "") == 1.0


class TestJsonValidity:
    def test_valid_json(self):
        assert compute_json_validity('{"key": "value"}') == 1.0

    def test_valid_json_array(self):
        assert compute_json_validity('[1, 2, 3]') == 1.0

    def test_invalid_json(self):
        assert compute_json_validity("not json at all") == 0.0

    def test_partial_json(self):
        assert compute_json_validity('{"key": "value"') == 0.0

    def test_empty_string(self):
        assert compute_json_validity("") == 0.0


class TestSchemaCompliance:
    def test_full_compliance(self):
        pred = '{"menu": [], "total": {}, "sub_total": {}}'
        expected = ["menu", "total", "sub_total"]
        assert compute_schema_compliance(pred, expected) == 1.0

    def test_partial_compliance(self):
        pred = '{"menu": [], "total": {}}'
        expected = ["menu", "total", "sub_total"]
        assert compute_schema_compliance(pred, expected) == pytest.approx(2 / 3, abs=0.01)

    def test_no_compliance(self):
        pred = '{"other": "field"}'
        expected = ["menu", "total"]
        assert compute_schema_compliance(pred, expected) == 0.0

    def test_invalid_json(self):
        assert compute_schema_compliance("not json", ["menu"]) == 0.0


class TestComputeAllMetrics:
    def test_aggregated_metrics(self):
        predictions = [
            {"flat": {"menu.0.nm": "Coffee", "total.total_price": "100"}, "raw": '{"menu": [{"nm": "Coffee"}], "total": {"total_price": "100"}}'},
            {"flat": {"menu.0.nm": "Tea"}, "raw": '{"menu": [{"nm": "Tea"}]}'},
        ]
        ground_truths = [
            {"flat": {"menu.0.nm": "Coffee", "total.total_price": "100"}, "raw": '{"menu": [{"nm": "Coffee"}], "total": {"total_price": "100"}}'},
            {"flat": {"menu.0.nm": "Tea", "total.total_price": "50"}, "raw": '{"menu": [{"nm": "Tea"}], "total": {"total_price": "50"}}'},
        ]
        result = compute_all_metrics(predictions, ground_truths)
        assert "field_f1_micro" in result
        assert "exact_match" in result
        assert "json_validity" in result
        assert 0 <= result["field_f1_micro"] <= 1.0
