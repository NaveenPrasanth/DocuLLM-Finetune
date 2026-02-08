"""Claude API-powered synthetic data generation for receipt extraction training.

Generates three types of synthetic data:
1. Instruction variants - paraphrased extraction instructions
2. Synthetic receipts - realistic receipt JSON following CORD schema
3. Error augmentations - OCR-like corruptions paired with corrections

Uses the Anthropic Python SDK with retry logic and rate limiting.
"""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Any

import anthropic

from src.config import get_env_var, load_base_config
from src.data.cord_loader import get_cord_schema

logger = logging.getLogger(__name__)

# ── Token cost estimation (approximate) ────────────────────────────────────

# Claude 3.5 Sonnet pricing (per million tokens, as of 2024)
INPUT_COST_PER_M = 3.00
OUTPUT_COST_PER_M = 15.00


def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate API cost in USD from token counts."""
    return (input_tokens / 1_000_000) * INPUT_COST_PER_M + (
        output_tokens / 1_000_000
    ) * OUTPUT_COST_PER_M


# ── Retry / rate-limit helpers ─────────────────────────────────────────────

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BASE_DELAY = 2.0  # seconds
_RATE_LIMIT_DELAY = 1.0  # seconds between requests


def _retry_api_call(
    fn,
    *,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    base_delay: float = _DEFAULT_BASE_DELAY,
):
    """Execute *fn* with exponential-backoff retries on transient errors.

    Handles ``anthropic.RateLimitError``, ``anthropic.APIConnectionError``,
    and ``anthropic.APIStatusError`` with status >= 500.

    Args:
        fn: Zero-argument callable that performs the API call.
        max_retries: Maximum number of retry attempts.
        base_delay: Initial backoff delay in seconds (doubled each retry).

    Returns:
        The return value of *fn* on success.

    Raises:
        The last caught exception after all retries are exhausted, or any
        non-retryable exception immediately.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except anthropic.RateLimitError as exc:
            last_exc = exc
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "Rate limited (attempt %d/%d). Retrying in %.1fs ...",
                attempt + 1,
                max_retries + 1,
                delay,
            )
            time.sleep(delay)
        except anthropic.APIConnectionError as exc:
            last_exc = exc
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "Connection error (attempt %d/%d): %s. Retrying in %.1fs ...",
                attempt + 1,
                max_retries + 1,
                exc,
                delay,
            )
            time.sleep(delay)
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                last_exc = exc
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Server error %d (attempt %d/%d). Retrying in %.1fs ...",
                    exc.status_code,
                    attempt + 1,
                    max_retries + 1,
                    delay,
                )
                time.sleep(delay)
            else:
                raise
    raise last_exc  # type: ignore[misc]


# ── Prompt templates ───────────────────────────────────────────────────────

_INSTRUCTION_VARIANT_PROMPT = """\
You are an expert prompt engineer. Given the following base instruction used to \
tell a vision-language model to extract receipt data, generate {num_variants} \
diverse paraphrased variants.

Each variant should express the same intent but use a DIFFERENT style. \
Cycle through these styles: formal, casual, concise (one-line), detailed \
(multi-sentence), technical, simple/beginner-friendly, bullet-point list, \
imperative, interrogative, and template-style with placeholders.

Base instruction:
"{base_instruction}"

Return a JSON array of strings. Each element is one variant instruction. \
Return ONLY the JSON array, no other text.
"""

_SYNTHETIC_RECEIPT_PROMPT = """\
You are a synthetic data generator for receipt OCR training. Generate \
{num_receipts} realistic receipt JSON objects following the CORD v2 schema below.

CORD Schema:
{schema}

Requirements for EACH receipt:
- Include 1-8 menu items with realistic item names (coffee, food, drinks, \
groceries, restaurant items, etc.) from diverse store types.
- Each menu item MUST have at least "nm" and "price". Include "cnt" and \
"unitprice" for about 60% of items.
- Prices should be realistic strings (e.g., "3.50", "12,000", "15.99"). \
Mix decimal formats.
- Include "total" with at least "total_price". Add "cashprice" and \
"changeprice" for ~40% of receipts.
- Include "sub_total" with "subtotal_price" and "tax_price" for ~70% of receipts.
- Totals should be mathematically plausible (sum of items + tax ~ total).
- Vary the receipt complexity: some simple (2-3 items), some complex (6-8 items).
- Do NOT include every field in every receipt; real receipts have sparse fields.

Return a JSON array of objects. Each object follows the CORD schema above. \
Return ONLY the JSON array, no other text.
"""

_ERROR_AUGMENTATION_PROMPT = """\
You are an OCR error simulation expert. Given the following list of clean \
receipt JSON objects, create {num_pairs} corrupted versions that simulate \
realistic OCR errors.

Clean receipts:
{receipts_json}

For EACH pair, apply 2-5 of these realistic OCR error types:
1. Character substitution: "l" -> "1", "O" -> "0", "S" -> "5", "B" -> "8", \
"rn" -> "m", "I" -> "l"
2. Truncation: cut off ends of long item names or addresses
3. Merged words: remove spaces between words ("Grand Total" -> "GrandTotal")
4. Split words: add spurious spaces ("Coffee" -> "Cof fee")
5. Missing characters: drop random characters from strings
6. Number corruption: swap digits ("3.50" -> "3.80"), add/remove decimal \
points ("12.50" -> "1250")
7. Case errors: random case changes ("TOTAL" -> "TotaL")
8. Extra noise characters: insert stray characters ("@", "#", random punctuation)
9. Field truncation: completely drop a field that was in the original
10. Partial field values: only keep first few characters of a value

Return a JSON array of objects. Each object has two keys:
- "corrupted": the OCR-corrupted receipt JSON
- "corrected": the original clean receipt JSON (copy from input)

Apply DIFFERENT error combinations to each pair. Make errors look natural, \
not random garbage. Return ONLY the JSON array, no other text.
"""


# ── Main generator class ──────────────────────────────────────────────────


class SyntheticGenerator:
    """Generate synthetic training data for receipt extraction using Claude.

    Args:
        api_key: Anthropic API key. Falls back to ``ANTHROPIC_API_KEY`` env var.
        model: Claude model identifier.
        max_retries: Max retry attempts for transient API errors.
        rate_limit_delay: Seconds to wait between successive API requests.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_retries: int = _DEFAULT_MAX_RETRIES,
        rate_limit_delay: float = _RATE_LIMIT_DELAY,
    ) -> None:
        resolved_key = api_key or get_env_var("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Anthropic API key is required. Pass it directly or set "
                "the ANTHROPIC_API_KEY environment variable."
            )

        self.client = anthropic.Anthropic(api_key=resolved_key)
        self.model = model
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay

        # Running totals for cost tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0

        logger.info("SyntheticGenerator initialised (model=%s)", self.model)

    # ── Internal helpers ───────────────────────────────────────────────

    def _call_claude(
        self,
        prompt: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.9,
    ) -> str:
        """Send a single prompt to Claude and return the text response.

        Applies retry logic and inter-request rate limiting.

        Args:
            prompt: The user message content.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            The text content from Claude's response.
        """

        def _do_call() -> anthropic.types.Message:
            return self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )

        response = _retry_api_call(
            _do_call,
            max_retries=self.max_retries,
        )

        # Track tokens
        self._total_input_tokens += response.usage.input_tokens
        self._total_output_tokens += response.usage.output_tokens

        logger.debug(
            "API call: in=%d out=%d tokens",
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

        # Rate-limit pause before the next request
        time.sleep(self.rate_limit_delay)

        # Extract text from content blocks
        text_parts = [
            block.text for block in response.content if block.type == "text"
        ]
        return "\n".join(text_parts)

    @staticmethod
    def _parse_json_response(text: str) -> Any:
        """Extract and parse a JSON array from Claude's response text.

        Handles cases where Claude wraps JSON in markdown code fences.

        Args:
            text: Raw response text.

        Returns:
            Parsed JSON (typically a list).

        Raises:
            ValueError: If no valid JSON array can be extracted.
        """
        cleaned = text.strip()

        # Strip markdown code fences if present
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first line (```json or ```) and last line (```)
            start = 1
            end = len(lines)
            for i in range(len(lines) - 1, 0, -1):
                if lines[i].strip().startswith("```"):
                    end = i
                    break
            cleaned = "\n".join(lines[start:end]).strip()

        # Find the JSON array boundaries
        arr_start = cleaned.find("[")
        arr_end = cleaned.rfind("]")

        if arr_start == -1 or arr_end == -1 or arr_end <= arr_start:
            raise ValueError(
                f"Could not find JSON array in response. "
                f"First 200 chars: {text[:200]}"
            )

        json_str = cleaned[arr_start : arr_end + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse JSON from response: {exc}. "
                f"First 200 chars of extracted JSON: {json_str[:200]}"
            ) from exc

    # ── Public generation methods ──────────────────────────────────────

    def generate_instruction_variants(
        self,
        base_instruction: str | None = None,
        num_variants: int = 20,
    ) -> list[str]:
        """Generate paraphrased instruction variants using Claude.

        Uses Claude to rewrite a receipt extraction instruction in multiple
        styles (formal, casual, concise, detailed, etc.) for instruction
        diversity in training.

        Args:
            base_instruction: The base instruction to paraphrase. Defaults to
                the standard receipt extraction instruction.
            num_variants: Number of variants to generate.

        Returns:
            List of instruction variant strings.

        Raises:
            ValueError: If the response cannot be parsed as a JSON string list.
            anthropic.APIError: On non-retryable API errors.
        """
        if base_instruction is None:
            base_instruction = (
                "Extract all information from this receipt image as structured "
                "JSON. Include menu items, totals, subtotals, and any other "
                "visible information. Return ONLY valid JSON."
            )

        prompt = _INSTRUCTION_VARIANT_PROMPT.format(
            num_variants=num_variants,
            base_instruction=base_instruction,
        )

        logger.info(
            "Generating %d instruction variants ...", num_variants
        )
        raw = self._call_claude(prompt, max_tokens=4096, temperature=0.9)
        variants = self._parse_json_response(raw)

        if not isinstance(variants, list):
            raise ValueError(
                f"Expected a JSON list of strings, got {type(variants).__name__}"
            )

        # Ensure all items are strings
        variants = [str(v) for v in variants]

        logger.info(
            "Generated %d instruction variants (requested %d)",
            len(variants),
            num_variants,
        )
        return variants

    def generate_synthetic_receipts(
        self,
        num_receipts: int = 100,
        schema: dict | None = None,
    ) -> list[dict]:
        """Generate realistic synthetic receipt JSON data via Claude.

        Generates receipts in batches to stay within token limits. Each
        receipt follows the CORD v2 schema with randomised items, prices,
        and totals.

        Args:
            num_receipts: Total number of receipts to generate.
            schema: CORD schema dict. Uses ``get_cord_schema()`` if *None*.

        Returns:
            List of dicts, each containing a ``"ground_truth_json"`` key
            with the JSON string of the receipt.

        Raises:
            ValueError: If any batch response cannot be parsed.
            anthropic.APIError: On non-retryable API errors.
        """
        if schema is None:
            schema = get_cord_schema()

        schema_str = json.dumps(schema, indent=2, ensure_ascii=False)

        # Generate in batches of up to 20 to keep responses manageable
        batch_size = min(20, num_receipts)
        all_receipts: list[dict] = []

        remaining = num_receipts
        batch_num = 0

        while remaining > 0:
            batch_count = min(batch_size, remaining)
            batch_num += 1

            prompt = _SYNTHETIC_RECEIPT_PROMPT.format(
                num_receipts=batch_count,
                schema=schema_str,
            )

            logger.info(
                "Generating receipt batch %d (%d receipts, %d remaining) ...",
                batch_num,
                batch_count,
                remaining,
            )

            raw = self._call_claude(prompt, max_tokens=8192, temperature=1.0)
            batch = self._parse_json_response(raw)

            if not isinstance(batch, list):
                raise ValueError(
                    f"Expected a JSON list of receipt dicts, "
                    f"got {type(batch).__name__}"
                )

            for receipt in batch:
                if not isinstance(receipt, dict):
                    logger.warning(
                        "Skipping non-dict receipt entry: %s",
                        type(receipt).__name__,
                    )
                    continue
                all_receipts.append({
                    "ground_truth_json": json.dumps(
                        receipt, ensure_ascii=False
                    ),
                })

            remaining -= batch_count

        logger.info(
            "Generated %d synthetic receipts (requested %d)",
            len(all_receipts),
            num_receipts,
        )
        return all_receipts

    def generate_error_augmentations(
        self,
        receipts: list[dict],
        num_pairs: int = 50,
    ) -> list[dict]:
        """Create OCR-error augmented receipt pairs for robustness training.

        Takes clean receipt dicts and uses Claude to produce realistic
        corruptions (character substitutions, truncations, merged words,
        etc.), returning paired corrupted/corrected data.

        Args:
            receipts: List of clean receipt dicts (each should have a
                ``"ground_truth_json"`` key or be a raw CORD receipt dict).
            num_pairs: Number of corrupted/corrected pairs to generate.

        Returns:
            List of dicts, each with ``"corrupted"`` and ``"corrected"``
            keys containing receipt JSON dicts.

        Raises:
            ValueError: If the response cannot be parsed.
            anthropic.APIError: On non-retryable API errors.
        """
        # Prepare clean receipt list for the prompt
        clean_receipts: list[dict] = []
        for r in receipts:
            if "ground_truth_json" in r:
                try:
                    clean_receipts.append(json.loads(r["ground_truth_json"]))
                except (json.JSONDecodeError, TypeError):
                    clean_receipts.append(r)
            elif "ground_truth" in r and isinstance(r["ground_truth"], dict):
                clean_receipts.append(r["ground_truth"])
            else:
                clean_receipts.append(r)

        if not clean_receipts:
            raise ValueError("No valid receipts provided for error augmentation.")

        # Generate in batches of up to 10 pairs per call
        batch_size = min(10, num_pairs)
        all_pairs: list[dict] = []
        remaining = num_pairs
        batch_num = 0

        while remaining > 0:
            batch_count = min(batch_size, remaining)
            batch_num += 1

            # Sample receipts to corrupt (with replacement if needed)
            if len(clean_receipts) >= batch_count:
                sampled = random.sample(clean_receipts, batch_count)
            else:
                sampled = random.choices(clean_receipts, k=batch_count)

            receipts_json = json.dumps(sampled, indent=2, ensure_ascii=False)
            prompt = _ERROR_AUGMENTATION_PROMPT.format(
                num_pairs=batch_count,
                receipts_json=receipts_json,
            )

            logger.info(
                "Generating error augmentation batch %d (%d pairs, %d remaining) ...",
                batch_num,
                batch_count,
                remaining,
            )

            raw = self._call_claude(prompt, max_tokens=8192, temperature=0.8)
            batch = self._parse_json_response(raw)

            if not isinstance(batch, list):
                raise ValueError(
                    f"Expected a JSON list of error pairs, "
                    f"got {type(batch).__name__}"
                )

            for pair in batch:
                if not isinstance(pair, dict):
                    logger.warning(
                        "Skipping non-dict error pair: %s",
                        type(pair).__name__,
                    )
                    continue
                if "corrupted" not in pair or "corrected" not in pair:
                    logger.warning(
                        "Skipping pair missing 'corrupted'/'corrected' keys: %s",
                        list(pair.keys()),
                    )
                    continue
                all_pairs.append({
                    "corrupted": pair["corrupted"],
                    "corrected": pair["corrected"],
                })

            remaining -= batch_count

        logger.info(
            "Generated %d error augmentation pairs (requested %d)",
            len(all_pairs),
            num_pairs,
        )
        return all_pairs

    # ── Convenience ────────────────────────────────────────────────────

    def generate_all(
        self,
        *,
        base_instruction: str | None = None,
        num_instruction_variants: int = 20,
        num_receipts: int = 100,
        receipt_schema: dict | None = None,
        num_error_pairs: int = 50,
    ) -> dict[str, Any]:
        """Run all three generation methods and return combined results.

        Args:
            base_instruction: Base instruction for variant generation.
            num_instruction_variants: Number of instruction variants.
            num_receipts: Number of synthetic receipts.
            receipt_schema: CORD schema override.
            num_error_pairs: Number of error augmentation pairs.

        Returns:
            Dictionary with keys ``"instruction_variants"``,
            ``"synthetic_receipts"``, ``"error_augmentations"``, and
            ``"cost_estimate"``.
        """
        logger.info(
            "Starting full synthetic data generation: "
            "%d instructions, %d receipts, %d error pairs",
            num_instruction_variants,
            num_receipts,
            num_error_pairs,
        )

        # Reset token counters for this run
        self._total_input_tokens = 0
        self._total_output_tokens = 0

        # 1. Instruction variants
        instruction_variants = self.generate_instruction_variants(
            base_instruction=base_instruction,
            num_variants=num_instruction_variants,
        )

        # 2. Synthetic receipts
        synthetic_receipts = self.generate_synthetic_receipts(
            num_receipts=num_receipts,
            schema=receipt_schema,
        )

        # 3. Error augmentations (use the synthetic receipts as input)
        error_augmentations = self.generate_error_augmentations(
            receipts=synthetic_receipts,
            num_pairs=num_error_pairs,
        )

        cost = _estimate_cost(
            self._total_input_tokens,
            self._total_output_tokens,
        )

        result = {
            "instruction_variants": instruction_variants,
            "synthetic_receipts": synthetic_receipts,
            "error_augmentations": error_augmentations,
            "cost_estimate": {
                "input_tokens": self._total_input_tokens,
                "output_tokens": self._total_output_tokens,
                "estimated_cost_usd": round(cost, 4),
            },
        }

        logger.info(
            "Synthetic generation complete. Tokens: in=%d, out=%d. "
            "Estimated cost: $%.4f",
            self._total_input_tokens,
            self._total_output_tokens,
            cost,
        )

        return result

    # ── Reporting ──────────────────────────────────────────────────────

    def get_usage_summary(self) -> dict[str, Any]:
        """Return cumulative token usage and cost estimate.

        Returns:
            Dictionary with ``input_tokens``, ``output_tokens``, and
            ``estimated_cost_usd``.
        """
        cost = _estimate_cost(
            self._total_input_tokens,
            self._total_output_tokens,
        )
        return {
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "estimated_cost_usd": round(cost, 4),
        }
