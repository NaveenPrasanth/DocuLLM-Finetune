"""Shared test fixtures for DocuMind."""

import json

import pytest
from PIL import Image


@pytest.fixture
def sample_image():
    """Create a simple test image."""
    return Image.new("RGB", (640, 480), color="white")


@pytest.fixture
def sample_cord_ground_truth():
    """Sample CORD ground truth JSON string."""
    gt = {
        "gt_parse": {
            "menu": [
                {"nm": "Americano", "unitprice": "4,500", "cnt": "1", "price": "4,500"},
                {"nm": "Latte", "unitprice": "5,000", "cnt": "2", "price": "10,000"},
            ],
            "total": {
                "total_price": "14,500",
                "cashprice": "20,000",
                "changeprice": "5,500",
            },
            "sub_total": {
                "subtotal_price": "14,500",
                "tax_price": "0",
            },
        }
    }
    return json.dumps(gt)


@pytest.fixture
def sample_cord_gt_parse():
    """Parsed CORD gt_parse dict."""
    return {
        "menu": [
            {"nm": "Americano", "unitprice": "4,500", "cnt": "1", "price": "4,500"},
            {"nm": "Latte", "unitprice": "5,000", "cnt": "2", "price": "10,000"},
        ],
        "total": {
            "total_price": "14,500",
            "cashprice": "20,000",
            "changeprice": "5,500",
        },
        "sub_total": {
            "subtotal_price": "14,500",
            "tax_price": "0",
        },
    }


@pytest.fixture
def sample_prediction_json():
    """Sample model prediction JSON string."""
    pred = {
        "menu": [
            {"nm": "Americano", "unitprice": "4,500", "cnt": "1", "price": "4,500"},
            {"nm": "Latte", "unitprice": "5,000", "cnt": "2", "price": "10,000"},
        ],
        "total": {
            "total_price": "14,500",
        },
        "sub_total": {
            "subtotal_price": "14,500",
        },
    }
    return json.dumps(pred)


@pytest.fixture
def sample_flat_ground_truth():
    """Flat field dict from CORD ground truth."""
    return {
        "menu.0.nm": "Americano",
        "menu.0.unitprice": "4,500",
        "menu.0.cnt": "1",
        "menu.0.price": "4,500",
        "menu.1.nm": "Latte",
        "menu.1.unitprice": "5,000",
        "menu.1.cnt": "2",
        "menu.1.price": "10,000",
        "total.total_price": "14,500",
        "total.cashprice": "20,000",
        "total.changeprice": "5,500",
        "sub_total.subtotal_price": "14,500",
        "sub_total.tax_price": "0",
    }


@pytest.fixture
def sample_flat_prediction():
    """Flat field dict from model prediction (partially correct)."""
    return {
        "menu.0.nm": "Americano",
        "menu.0.unitprice": "4,500",
        "menu.0.cnt": "1",
        "menu.0.price": "4,500",
        "menu.1.nm": "Latte",
        "menu.1.unitprice": "5,000",
        "menu.1.cnt": "2",
        "menu.1.price": "10,000",
        "total.total_price": "14,500",
        "sub_total.subtotal_price": "14,500",
    }
