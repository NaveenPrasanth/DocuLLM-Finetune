"""Prompt strategy implementations for document extraction.

Provides 7 prompt strategies for evaluating extraction quality on the Qwen2-VL
base model:
  - Zero-shot: basic, detailed, structured (with JSON schema)
  - Few-shot: 2-example and 5-example variants
  - Chain-of-thought: step-by-step reasoning and self-verification

Each strategy produces ChatML-format messages compatible with Qwen2-VL inference.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from PIL import Image

from src.data.cord_loader import get_cord_schema
from src.data.format_converter import sample_to_inference_chatml
from src.prompts.templates import ReceiptExtractionTemplate

logger = logging.getLogger(__name__)


# ── Strategy Registry ────────────────────────────────────────────────────────

_STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {}


def register_strategy(name: str):
    """Decorator to register a strategy class by name."""

    def decorator(cls: type[BaseStrategy]):
        _STRATEGY_REGISTRY[name] = cls
        return cls

    return decorator


def get_strategy(name: str, **kwargs: Any) -> BaseStrategy:
    """Retrieve a registered strategy by name.

    Args:
        name: Registered strategy name.
        **kwargs: Arguments forwarded to the strategy constructor.

    Returns:
        Instantiated strategy.

    Raises:
        KeyError: If the strategy name is not registered.
    """
    if name not in _STRATEGY_REGISTRY:
        available = ", ".join(sorted(_STRATEGY_REGISTRY.keys()))
        raise KeyError(
            f"Strategy '{name}' not found. Available: {available}"
        )
    return _STRATEGY_REGISTRY[name](**kwargs)


def list_strategies() -> list[str]:
    """Return sorted list of all registered strategy names."""
    return sorted(_STRATEGY_REGISTRY.keys())


def get_all_strategies(**kwargs: Any) -> list[BaseStrategy]:
    """Instantiate all registered strategies.

    Args:
        **kwargs: Common keyword arguments forwarded to constructors that
            accept them (e.g. ``example_samples`` for few-shot strategies).

    Returns:
        List of all strategy instances.
    """
    strategies = []
    for name in sorted(_STRATEGY_REGISTRY.keys()):
        cls = _STRATEGY_REGISTRY[name]
        # Only pass kwargs that the constructor accepts
        import inspect

        sig = inspect.signature(cls.__init__)
        valid_kwargs = {
            k: v for k, v in kwargs.items() if k in sig.parameters
        }
        strategies.append(cls(**valid_kwargs))
    return strategies


# ── Schema Helpers ───────────────────────────────────────────────────────────

# Concise CORD schema for use in prompts (matches templates.py)
CORD_PROMPT_SCHEMA: dict[str, Any] = ReceiptExtractionTemplate.CORD_PROMPT_SCHEMA

CORD_SCHEMA_STR: str = json.dumps(CORD_PROMPT_SCHEMA, indent=2)


def _format_example(sample: dict[str, Any], index: int) -> str:
    """Format a single sample as a text example for few-shot prompts.

    Args:
        sample: Dataset sample dict with ground_truth key.
        index: 1-based example number.

    Returns:
        Formatted example string.
    """
    gt = sample.get("ground_truth", {})
    gt_json = json.dumps(gt, indent=2, ensure_ascii=False)
    return f"Example {index}:\n{gt_json}"


# ── Base Strategy ────────────────────────────────────────────────────────────


class BaseStrategy(ABC):
    """Abstract base class for prompt strategies.

    Subclasses must implement build_prompt() to produce the instruction text
    and get_name() to return a unique strategy identifier.
    """

    @abstractmethod
    def build_prompt(self, sample: dict[str, Any]) -> str:
        """Build the prompt instruction text for a given sample.

        Args:
            sample: Dataset sample dict containing at minimum an 'image' key.

        Returns:
            Instruction text string (does not include the image).
        """
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Return the unique name of this strategy."""
        ...

    def build_messages(self, sample: dict[str, Any]) -> list[dict[str, Any]]:
        """Build ChatML-format messages for Qwen2-VL inference.

        Constructs a single user message containing the image and the
        instruction text from build_prompt().

        Args:
            sample: Dataset sample dict with 'image' (PIL Image).

        Returns:
            List of ChatML message dicts suitable for Qwen2-VL.
        """
        instruction = self.build_prompt(sample)
        image: Image.Image = sample["image"]
        return sample_to_inference_chatml(image=image, instruction=instruction)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.get_name()}')"


# ── Zero-Shot Strategies ────────────────────────────────────────────────────


@register_strategy("zero_shot_basic")
class ZeroShotBasic(BaseStrategy):
    """Minimal zero-shot instruction with no schema guidance.

    Provides only a brief instruction to extract receipt fields as JSON,
    relying entirely on the model's pre-trained knowledge.
    """

    def get_name(self) -> str:
        return "zero_shot_basic"

    def build_prompt(self, sample: dict[str, Any]) -> str:
        return "Extract all fields from this receipt image as JSON."


@register_strategy("zero_shot_detailed")
class ZeroShotDetailed(BaseStrategy):
    """Detailed zero-shot instruction with field categories.

    Describes the expected output categories (menu, total, sub_total)
    without providing the exact JSON schema.
    """

    def get_name(self) -> str:
        return "zero_shot_detailed"

    def build_prompt(self, sample: dict[str, Any]) -> str:
        return (
            "You are a document extraction specialist. Analyze this receipt "
            "image and extract all information into a structured JSON format.\n"
            "\n"
            "Extract the following categories:\n"
            "- menu: item names, quantities, unit prices, total prices\n"
            "- total: total price, cash, change, credit card amounts\n"
            "- sub_total: subtotal, discounts, tax, service charges\n"
            "- etc: any other relevant information\n"
            "\n"
            "Return ONLY valid JSON, no explanations."
        )


@register_strategy("zero_shot_structured")
class ZeroShotStructured(BaseStrategy):
    """Schema-guided zero-shot instruction.

    Includes the exact JSON schema the model should produce, along with
    explicit formatting rules.
    """

    def get_name(self) -> str:
        return "zero_shot_structured"

    def build_prompt(self, sample: dict[str, Any]) -> str:
        return (
            "Extract receipt information from this image into the exact "
            "JSON schema below.\n"
            "\n"
            f"Schema:\n{CORD_SCHEMA_STR}\n"
            "\n"
            "Rules:\n"
            "1. Use null for fields not present in the image\n"
            "2. Preserve exact text as shown (including currency symbols)\n"
            "3. Menu items should be a list of objects\n"
            "4. Return ONLY the JSON object, no markdown or explanations"
        )


# ── Few-Shot Strategies ─────────────────────────────────────────────────────


@register_strategy("few_shot_2")
class FewShotStrategy2(BaseStrategy):
    """Two-example few-shot strategy.

    Provides 2 representative extraction examples before the query,
    helping the model learn the expected output format and level of detail.
    """

    def __init__(self, example_samples: list[dict[str, Any]] | None = None):
        """Initialize with optional example samples.

        Args:
            example_samples: List of dataset samples to use as few-shot
                examples. If None, placeholder examples will be used.
        """
        self.num_examples = 2
        self.example_samples = example_samples

    def get_name(self) -> str:
        return "few_shot_2"

    def _get_examples_text(self) -> str:
        """Build formatted examples text from example_samples or defaults."""
        if self.example_samples and len(self.example_samples) >= self.num_examples:
            examples = self.example_samples[: self.num_examples]
            parts = [
                _format_example(ex, i + 1) for i, ex in enumerate(examples)
            ]
            return "\n\n".join(parts)

        # Fallback: representative synthetic examples
        return self._default_examples()

    def _default_examples(self) -> str:
        """Return default synthetic examples for CORD receipts."""
        example_1 = {
            "menu": [
                {"nm": "Americano", "unitprice": "4,500", "cnt": "1", "price": "4,500"},
                {"nm": "Latte", "unitprice": "5,000", "cnt": "2", "price": "10,000"},
            ],
            "total": {"total_price": "14,500", "cashprice": "20,000", "changeprice": "5,500"},
            "sub_total": {"subtotal_price": "14,500", "tax_price": None, "discount_price": None},
        }
        example_2 = {
            "menu": [
                {"nm": "Croissant", "unitprice": "3,000", "cnt": "1", "price": "3,000"},
            ],
            "total": {"total_price": "3,000", "cashprice": "3,000", "changeprice": "0"},
            "sub_total": {"subtotal_price": "3,000", "tax_price": None, "discount_price": None},
        }
        ex1 = f"Example 1:\n{json.dumps(example_1, indent=2)}"
        ex2 = f"Example 2:\n{json.dumps(example_2, indent=2)}"
        return f"{ex1}\n\n{ex2}"

    def build_prompt(self, sample: dict[str, Any]) -> str:
        examples_text = self._get_examples_text()
        return (
            "Extract receipt information from images as structured JSON.\n"
            "\n"
            "Here are examples of correct extractions:\n"
            "\n"
            f"{examples_text}\n"
            "\n"
            "Now extract from the current image using the same format. "
            "Return ONLY valid JSON."
        )


@register_strategy("few_shot_5")
class FewShotStrategy5(BaseStrategy):
    """Five-example few-shot strategy.

    Provides 5 diverse extraction examples covering different receipt
    complexities (simple, multi-item, discounts, tax, etc.).
    """

    def __init__(self, example_samples: list[dict[str, Any]] | None = None):
        """Initialize with optional example samples.

        Args:
            example_samples: List of dataset samples to use as few-shot
                examples. If None, placeholder examples will be used.
        """
        self.num_examples = 5
        self.example_samples = example_samples

    def get_name(self) -> str:
        return "few_shot_5"

    def _get_examples_text(self) -> str:
        """Build formatted examples text from example_samples or defaults."""
        if self.example_samples and len(self.example_samples) >= self.num_examples:
            examples = self.example_samples[: self.num_examples]
            parts = [
                _format_example(ex, i + 1) for i, ex in enumerate(examples)
            ]
            return "\n\n".join(parts)

        return self._default_examples()

    def _default_examples(self) -> str:
        """Return 5 default synthetic examples covering diverse receipt types."""
        examples = [
            {
                "menu": [
                    {"nm": "Americano", "unitprice": "4,500", "cnt": "1", "price": "4,500"},
                    {"nm": "Latte", "unitprice": "5,000", "cnt": "2", "price": "10,000"},
                ],
                "total": {"total_price": "14,500", "cashprice": "20,000", "changeprice": "5,500"},
                "sub_total": {"subtotal_price": "14,500", "tax_price": None, "discount_price": None},
            },
            {
                "menu": [
                    {"nm": "Croissant", "unitprice": "3,000", "cnt": "1", "price": "3,000"},
                ],
                "total": {"total_price": "3,000", "cashprice": "3,000", "changeprice": "0"},
                "sub_total": {"subtotal_price": "3,000", "tax_price": None, "discount_price": None},
            },
            {
                "menu": [
                    {"nm": "Green Tea", "unitprice": "4,000", "cnt": "1", "price": "4,000"},
                    {"nm": "Cake Slice", "unitprice": "5,500", "cnt": "1", "price": "5,500"},
                    {"nm": "Cookie", "unitprice": "2,000", "cnt": "3", "price": "6,000"},
                ],
                "total": {"total_price": "15,500", "cashprice": None, "changeprice": None},
                "sub_total": {
                    "subtotal_price": "15,500",
                    "tax_price": None,
                    "discount_price": None,
                },
            },
            {
                "menu": [
                    {"nm": "Set Menu A", "unitprice": "12,000", "cnt": "1", "price": "12,000"},
                    {"nm": "Extra Side", "unitprice": "3,000", "cnt": "1", "price": "3,000"},
                ],
                "total": {"total_price": "13,500", "cashprice": "15,000", "changeprice": "1,500"},
                "sub_total": {
                    "subtotal_price": "15,000",
                    "tax_price": None,
                    "discount_price": "1,500",
                },
            },
            {
                "menu": [
                    {"nm": "Espresso", "unitprice": "3,500", "cnt": "2", "price": "7,000"},
                ],
                "total": {"total_price": "7,700", "cashprice": "10,000", "changeprice": "2,300"},
                "sub_total": {
                    "subtotal_price": "7,000",
                    "tax_price": "700",
                    "discount_price": None,
                },
            },
        ]
        parts = []
        for i, ex in enumerate(examples):
            parts.append(f"Example {i + 1}:\n{json.dumps(ex, indent=2)}")
        return "\n\n".join(parts)

    def build_prompt(self, sample: dict[str, Any]) -> str:
        examples_text = self._get_examples_text()
        return (
            "Extract receipt information from images as structured JSON.\n"
            "\n"
            "Here are examples of correct extractions:\n"
            "\n"
            f"{examples_text}\n"
            "\n"
            "Now extract from the current image using the same format. "
            "Return ONLY valid JSON."
        )


# ── Chain-of-Thought Strategies ─────────────────────────────────────────────


@register_strategy("cot_step_by_step")
class ChainOfThoughtStepByStep(BaseStrategy):
    """Step-by-step chain-of-thought reasoning strategy.

    Instructs the model to work through the receipt extraction in explicit
    stages: layout identification, menu items, subtotals, totals, and
    final JSON compilation.
    """

    def get_name(self) -> str:
        return "cot_step_by_step"

    def build_prompt(self, sample: dict[str, Any]) -> str:
        return (
            "Extract all receipt information from this image step by step.\n"
            "\n"
            "Step 1: Identify the receipt layout and sections\n"
            "Step 2: Read each menu item (name, quantity, price)\n"
            "Step 3: Find subtotals and discounts\n"
            "Step 4: Find the total amount and payment details\n"
            "Step 5: Compile everything into this JSON format:\n"
            "\n"
            f"{CORD_SCHEMA_STR}\n"
            "\n"
            'Think through each step, then provide the final JSON after "RESULT:".'
        )


@register_strategy("cot_self_verify")
class ChainOfThoughtSelfVerify(BaseStrategy):
    """Self-verification chain-of-thought strategy.

    Instructs the model to first extract all fields, then verify the
    extraction by checking arithmetic consistency (item prices vs subtotal,
    subtotal + tax vs total) and completeness.
    """

    def get_name(self) -> str:
        return "cot_self_verify"

    def build_prompt(self, sample: dict[str, Any]) -> str:
        return (
            "Extract all receipt information from this image.\n"
            "\n"
            "First, extract all fields into JSON format:\n"
            f"{CORD_SCHEMA_STR}\n"
            "\n"
            "Then verify your extraction:\n"
            "- Do the item prices sum to the subtotal?\n"
            "- Does subtotal + tax = total?\n"
            "- Are all visible items captured?\n"
            "- Are currency amounts formatted correctly?\n"
            "\n"
            "If you find errors, correct them. Return the final verified "
            'JSON after "RESULT:".'
        )
