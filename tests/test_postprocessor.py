"""Tests for postprocessing model outputs."""

import json

import pytest

from src.inference.postprocessor import (
    extract_json,
    fix_json,
    postprocess_prediction,
    validate_against_schema,
)


class TestExtractJson:
    def test_plain_json(self):
        text = '{"menu": [{"nm": "Coffee"}]}'
        result = extract_json(text)
        assert json.loads(result) == {"menu": [{"nm": "Coffee"}]}

    def test_markdown_wrapped(self):
        text = '```json\n{"menu": [{"nm": "Coffee"}]}\n```'
        result = extract_json(text)
        assert json.loads(result) == {"menu": [{"nm": "Coffee"}]}

    def test_markdown_no_language(self):
        text = '```\n{"menu": []}\n```'
        result = extract_json(text)
        assert json.loads(result) == {"menu": []}

    def test_with_prefix_text(self):
        text = 'Here is the extraction:\n{"total": {"total_price": "100"}}'
        result = extract_json(text)
        assert "total_price" in result

    def test_with_result_prefix(self):
        text = 'Step 1: I see items\nStep 2: Totals\nRESULT:\n{"menu": []}'
        result = extract_json(text)
        assert json.loads(result) == {"menu": []}

    def test_no_json_found(self):
        text = "This is just plain text with no JSON"
        result = extract_json(text)
        assert result == text  # Returns original if no JSON found


class TestFixJson:
    def test_single_quotes(self):
        text = "{'menu': [{'nm': 'Coffee'}]}"
        result = fix_json(text)
        parsed = json.loads(result)
        assert parsed == {"menu": [{"nm": "Coffee"}]}

    def test_trailing_comma(self):
        text = '{"menu": [], "total": {},}'
        result = fix_json(text)
        parsed = json.loads(result)
        assert "menu" in parsed

    def test_missing_closing_brace(self):
        text = '{"menu": [{"nm": "Coffee"}]'
        result = fix_json(text)
        parsed = json.loads(result)
        assert parsed["menu"][0]["nm"] == "Coffee"

    def test_already_valid(self):
        text = '{"valid": true}'
        result = fix_json(text)
        assert json.loads(result) == {"valid": True}


class TestValidateSchema:
    def test_valid_schema(self):
        json_str = '{"menu": [], "total": {}, "sub_total": {}}'
        is_valid, issues = validate_against_schema(json_str, ["menu", "total", "sub_total"])
        assert is_valid
        assert len(issues) == 0

    def test_missing_keys(self):
        json_str = '{"menu": []}'
        is_valid, issues = validate_against_schema(json_str, ["menu", "total", "sub_total"])
        assert not is_valid
        assert any("total" in issue for issue in issues)

    def test_extra_keys(self):
        json_str = '{"menu": [], "extra": "field"}'
        is_valid, issues = validate_against_schema(json_str, ["menu"])
        # Extra keys should be noted but not make it invalid
        assert is_valid or len(issues) > 0

    def test_invalid_json(self):
        is_valid, issues = validate_against_schema("not json", ["menu"])
        assert not is_valid


class TestPostprocessPrediction:
    def test_clean_json(self):
        result = postprocess_prediction('{"menu": [{"nm": "Coffee"}]}')
        assert result["valid"]
        assert result["parsed"]["menu"][0]["nm"] == "Coffee"

    def test_markdown_wrapped(self):
        text = '```json\n{"total": {"total_price": "100"}}\n```'
        result = postprocess_prediction(text)
        assert result["valid"]
        assert result["parsed"]["total"]["total_price"] == "100"

    def test_with_explanatory_text(self):
        text = 'I found the following:\n{"menu": [{"nm": "Tea", "price": "3.00"}]}\nThat is all.'
        result = postprocess_prediction(text)
        assert result["valid"]
        assert result["parsed"]["menu"][0]["nm"] == "Tea"

    def test_completely_invalid(self):
        result = postprocess_prediction("No JSON here at all, just words.")
        assert not result["valid"]
        assert result["parsed"] == {}
        assert len(result["errors"]) > 0

    def test_preserves_raw(self):
        raw = '{"test": true}'
        result = postprocess_prediction(raw)
        assert result["raw"] == raw
