"""Tests for data loading and parsing."""

import json

import pytest

from src.data.cord_loader import (
    flatten_cord_fields,
    get_cord_schema,
    parse_cord_ground_truth,
)


class TestParseGroundTruth:
    def test_parse_valid_json(self, sample_cord_ground_truth):
        result = parse_cord_ground_truth(sample_cord_ground_truth)
        assert "menu" in result
        assert "total" in result
        assert len(result["menu"]) == 2

    def test_parse_invalid_json(self):
        result = parse_cord_ground_truth("not valid json")
        assert result == {}

    def test_parse_none(self):
        result = parse_cord_ground_truth(None)
        assert result == {}

    def test_parse_empty_string(self):
        result = parse_cord_ground_truth("")
        assert result == {}

    def test_parse_missing_gt_parse(self):
        result = parse_cord_ground_truth('{"other_key": "value"}')
        assert result == {}


class TestFlattenFields:
    def test_flatten_menu_items(self, sample_cord_gt_parse):
        flat = flatten_cord_fields(sample_cord_gt_parse)
        assert "menu.0.nm" in flat
        assert flat["menu.0.nm"] == "Americano"
        assert "menu.1.nm" in flat
        assert flat["menu.1.nm"] == "Latte"

    def test_flatten_total(self, sample_cord_gt_parse):
        flat = flatten_cord_fields(sample_cord_gt_parse)
        assert "total.total_price" in flat
        assert flat["total.total_price"] == "14,500"
        assert "total.cashprice" in flat

    def test_flatten_subtotal(self, sample_cord_gt_parse):
        flat = flatten_cord_fields(sample_cord_gt_parse)
        assert "sub_total.subtotal_price" in flat

    def test_flatten_empty(self):
        flat = flatten_cord_fields({})
        assert flat == {}

    def test_flatten_skips_empty_values(self):
        gt = {"total": {"total_price": "100", "cashprice": "", "changeprice": None}}
        flat = flatten_cord_fields(gt)
        assert "total.total_price" in flat
        assert "total.cashprice" not in flat
        assert "total.changeprice" not in flat

    def test_total_field_count(self, sample_cord_gt_parse):
        flat = flatten_cord_fields(sample_cord_gt_parse)
        # 4 fields per menu item * 2 items + 3 total + 2 sub_total = 13
        assert len(flat) == 13


class TestCordSchema:
    def test_schema_has_all_superclasses(self):
        schema = get_cord_schema()
        assert "menu" in schema
        assert "total" in schema
        assert "sub_total" in schema
        assert "void_menu" in schema

    def test_schema_menu_is_list(self):
        schema = get_cord_schema()
        assert isinstance(schema["menu"], list)
        assert "nm" in schema["menu"][0]

    def test_schema_total_is_dict(self):
        schema = get_cord_schema()
        assert isinstance(schema["total"], dict)
        assert "total_price" in schema["total"]
