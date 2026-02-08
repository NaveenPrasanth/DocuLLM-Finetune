"""Prompt template classes with variable substitution and registry pattern.

Provides reusable templates for document extraction prompts, with dataset-specific
subclasses for CORD (receipts) and FUNSD (forms).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.data.cord_loader import get_cord_schema
from src.data.funsd_loader import get_funsd_schema

logger = logging.getLogger(__name__)


# ── Template Registry ────────────────────────────────────────────────────────

_TEMPLATE_REGISTRY: dict[str, type[PromptTemplate]] = {}


def register_template(name: str):
    """Decorator to register a template class by name."""

    def decorator(cls: type[PromptTemplate]):
        _TEMPLATE_REGISTRY[name] = cls
        return cls

    return decorator


def get_template(name: str, **kwargs: Any) -> PromptTemplate:
    """Retrieve a registered template by name.

    Args:
        name: Registered template name.
        **kwargs: Additional keyword arguments passed to the constructor.

    Returns:
        Instantiated PromptTemplate subclass.

    Raises:
        KeyError: If the template name is not registered.
    """
    if name not in _TEMPLATE_REGISTRY:
        available = ", ".join(sorted(_TEMPLATE_REGISTRY.keys()))
        raise KeyError(
            f"Template '{name}' not found. Available templates: {available}"
        )
    return _TEMPLATE_REGISTRY[name](**kwargs)


def list_templates() -> list[str]:
    """Return sorted list of all registered template names."""
    return sorted(_TEMPLATE_REGISTRY.keys())


# ── Base Template ────────────────────────────────────────────────────────────


class PromptTemplate:
    """Base prompt template with variable substitution.

    Uses Python str.format() for variable interpolation. Variables are
    provided as keyword arguments to render().

    Attributes:
        template: Format string containing {variable} placeholders.
        variables: Default values for template variables.
        name: Human-readable template name.
        description: Optional description of what this template does.
    """

    def __init__(
        self,
        template: str,
        variables: dict[str, Any] | None = None,
        name: str = "base",
        description: str = "",
    ):
        self.template = template
        self.variables = variables or {}
        self.name = name
        self.description = description

    def render(self, **kwargs: Any) -> str:
        """Render the template by substituting variables.

        Merges default variables with provided kwargs. Kwargs take precedence
        over defaults.

        Args:
            **kwargs: Variable values to substitute into the template.

        Returns:
            Rendered prompt string.

        Raises:
            KeyError: If a required template variable is missing.
        """
        merged = {**self.variables, **kwargs}
        try:
            return self.template.format(**merged)
        except KeyError as e:
            available = ", ".join(sorted(merged.keys()))
            raise KeyError(
                f"Missing template variable {e}. "
                f"Available variables: {available}"
            ) from e

    def get_required_variables(self) -> list[str]:
        """Extract variable names from the template string.

        Returns:
            List of variable names found in {variable} placeholders.
        """
        import string

        formatter = string.Formatter()
        return [
            field_name
            for _, field_name, _, _ in formatter.parse(self.template)
            if field_name is not None
        ]

    def __repr__(self) -> str:
        req_vars = self.get_required_variables()
        return (
            f"PromptTemplate(name='{self.name}', "
            f"variables={req_vars})"
        )


# ── CORD Receipt Template ───────────────────────────────────────────────────


@register_template("receipt_extraction")
class ReceiptExtractionTemplate(PromptTemplate):
    """Template specialized for CORD receipt extraction.

    Pre-configured with the CORD JSON schema and receipt-specific defaults.
    """

    # Simplified CORD schema used inside prompts (concise for token efficiency)
    CORD_PROMPT_SCHEMA: dict[str, Any] = {
        "menu": [
            {
                "nm": "item name",
                "unitprice": "price",
                "cnt": "qty",
                "price": "total",
            }
        ],
        "total": {
            "total_price": "amount",
            "cashprice": "cash",
            "changeprice": "change",
        },
        "sub_total": {
            "subtotal_price": "subtotal",
            "tax_price": "tax",
            "discount_price": "discount",
        },
    }

    DEFAULT_TEMPLATE = (
        "Extract receipt information from this image into the exact JSON schema below.\n"
        "\n"
        "Schema:\n"
        "{schema}\n"
        "\n"
        "Rules:\n"
        "1. Use null for fields not present in the image\n"
        "2. Preserve exact text as shown (including currency symbols)\n"
        "3. Menu items should be a list of objects\n"
        "4. Return ONLY the JSON object, no markdown or explanations"
    )

    def __init__(
        self,
        template: str | None = None,
        variables: dict[str, Any] | None = None,
        name: str = "receipt_extraction",
        description: str = "CORD receipt extraction with schema guidance",
    ):
        schema_str = json.dumps(self.CORD_PROMPT_SCHEMA, indent=2)
        defaults = {"schema": schema_str}
        if variables:
            defaults.update(variables)

        super().__init__(
            template=template or self.DEFAULT_TEMPLATE,
            variables=defaults,
            name=name,
            description=description,
        )

    @classmethod
    def get_full_schema(cls) -> dict[str, Any]:
        """Return the full CORD schema from the data loader."""
        return get_cord_schema()

    @classmethod
    def get_prompt_schema(cls) -> dict[str, Any]:
        """Return the concise schema used in prompts."""
        return cls.CORD_PROMPT_SCHEMA


# ── FUNSD Form Template ─────────────────────────────────────────────────────


@register_template("form_extraction")
class FormExtractionTemplate(PromptTemplate):
    """Template specialized for FUNSD form key-value extraction.

    Pre-configured for extracting key-value pairs from scanned form documents.
    """

    FUNSD_PROMPT_SCHEMA: dict[str, str] = {
        "<field_name>": "<field_value>",
    }

    DEFAULT_TEMPLATE = (
        "Extract all key-value pairs from this form document image.\n"
        "\n"
        "Expected output format:\n"
        "{schema}\n"
        "\n"
        "Instructions:\n"
        "1. Identify all form fields (questions/labels) and their corresponding values\n"
        "2. Map each field name to its value as a JSON object\n"
        "3. Use the exact text as written in the form\n"
        "4. If a field has no value, use an empty string\n"
        "5. Return ONLY the JSON object, no markdown or explanations"
    )

    def __init__(
        self,
        template: str | None = None,
        variables: dict[str, Any] | None = None,
        name: str = "form_extraction",
        description: str = "FUNSD form key-value extraction",
    ):
        schema_str = json.dumps(self.FUNSD_PROMPT_SCHEMA, indent=2)
        defaults = {"schema": schema_str}
        if variables:
            defaults.update(variables)

        super().__init__(
            template=template or self.DEFAULT_TEMPLATE,
            variables=defaults,
            name=name,
            description=description,
        )

    @classmethod
    def get_schema(cls) -> dict[str, str]:
        """Return the FUNSD schema from the data loader."""
        return get_funsd_schema()


# ── Convenience Constructors ─────────────────────────────────────────────────


def receipt_template(**kwargs: Any) -> ReceiptExtractionTemplate:
    """Create a ReceiptExtractionTemplate with optional overrides."""
    return ReceiptExtractionTemplate(**kwargs)


def form_template(**kwargs: Any) -> FormExtractionTemplate:
    """Create a FormExtractionTemplate with optional overrides."""
    return FormExtractionTemplate(**kwargs)
