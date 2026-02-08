"""JSON extraction and cleanup from model outputs.

Handles common model output issues: markdown code blocks, explanatory text,
invalid JSON syntax, missing brackets, and schema validation.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def extract_json(text: str) -> str:
    """Extract JSON from model output that may contain surrounding text.

    Handles these common patterns:
    - Markdown code blocks: ```json ... ``` or ``` ... ```
    - "RESULT:" or "Output:" prefixes
    - Explanatory text before/after the JSON
    - Multiple JSON objects (returns the first complete one)

    Args:
        text: Raw model output string.

    Returns:
        Extracted JSON string, or the original text if no JSON is found.
    """
    text = text.strip()

    # 1. Check for markdown code blocks
    # Match ```json ... ``` or ``` ... ```
    code_block_pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    match = re.search(code_block_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 2. Strip common prefixes
    prefix_patterns = [
        r"^(?:RESULT|Result|OUTPUT|Output|JSON|json)\s*:\s*",
        r"^(?:Here(?:'s| is) (?:the )?(?:extracted )?(?:JSON|result|output)\s*:?\s*)",
        r"^(?:The (?:extracted )?(?:JSON|result|output) is\s*:?\s*)",
    ]
    cleaned = text
    for pattern in prefix_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    # 3. Find the first complete JSON object or array
    # Look for outermost { ... } or [ ... ]
    json_str = _find_json_bounds(cleaned)
    if json_str:
        return json_str

    # 4. Fallback: try the cleaned text directly
    return cleaned


def _find_json_bounds(text: str) -> str | None:
    """Find the first balanced JSON object or array in text.

    Uses bracket counting to find matching open/close braces.

    Args:
        text: Text potentially containing a JSON object or array.

    Returns:
        The extracted JSON string, or None if not found.
    """
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start_idx = text.find(start_char)
        if start_idx == -1:
            continue

        depth = 0
        in_string = False
        escape_next = False

        for i in range(start_idx, len(text)):
            char = text[i]

            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == start_char:
                depth += 1
            elif char == end_char:
                depth -= 1
                if depth == 0:
                    return text[start_idx : i + 1]

    return None


def fix_json(text: str) -> str:
    """Attempt to fix common JSON syntax issues.

    Fixes applied in order:
    1. Single quotes -> double quotes (outside of already double-quoted strings)
    2. Trailing commas before } or ]
    3. Missing closing braces/brackets
    4. Unescaped newlines within string values
    5. Python-style None/True/False -> null/true/false

    Args:
        text: Potentially malformed JSON string.

    Returns:
        Fixed JSON string (may still be invalid in edge cases).
    """
    if not text.strip():
        return text

    fixed = text.strip()

    # 1. Replace Python literals with JSON equivalents
    # Only outside of quoted strings (simple heuristic)
    fixed = _replace_outside_strings(fixed, "None", "null")
    fixed = _replace_outside_strings(fixed, "True", "true")
    fixed = _replace_outside_strings(fixed, "False", "false")

    # 2. Single quotes -> double quotes (careful approach)
    # Only if the text doesn't already have proper double-quoted strings
    if not _has_valid_json_strings(fixed):
        fixed = _single_to_double_quotes(fixed)

    # 3. Remove trailing commas before closing brackets
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)

    # 4. Fix unescaped newlines within string values
    fixed = _fix_unescaped_newlines(fixed)

    # 5. Attempt to close unclosed braces/brackets
    fixed = _close_unclosed_brackets(fixed)

    return fixed


def _replace_outside_strings(text: str, old: str, new: str) -> str:
    """Replace occurrences of `old` with `new` only when outside JSON strings."""
    result = []
    in_string = False
    escape_next = False
    i = 0

    while i < len(text):
        if escape_next:
            result.append(text[i])
            escape_next = False
            i += 1
            continue

        if text[i] == "\\":
            result.append(text[i])
            escape_next = True
            i += 1
            continue

        if text[i] == '"':
            in_string = not in_string
            result.append(text[i])
            i += 1
            continue

        if not in_string and text[i:i + len(old)] == old:
            # Check word boundaries
            before_ok = (i == 0 or not text[i - 1].isalnum())
            after_ok = (i + len(old) >= len(text) or not text[i + len(old)].isalnum())
            if before_ok and after_ok:
                result.append(new)
                i += len(old)
                continue

        result.append(text[i])
        i += 1

    return "".join(result)


def _has_valid_json_strings(text: str) -> bool:
    """Check if the text appears to use double-quoted strings properly."""
    # Simple heuristic: if we see "key": patterns, assume double quotes are used
    return bool(re.search(r'"[^"]*"\s*:', text))


def _single_to_double_quotes(text: str) -> str:
    """Convert single-quoted strings to double-quoted strings.

    Handles nested quotes by escaping inner double quotes.
    """
    result = []
    in_single_string = False
    in_double_string = False
    escape_next = False

    for i, char in enumerate(text):
        if escape_next:
            result.append(char)
            escape_next = False
            continue

        if char == "\\":
            result.append(char)
            escape_next = True
            continue

        if char == '"' and not in_single_string:
            in_double_string = not in_double_string
            result.append(char)
            continue

        if char == "'" and not in_double_string:
            if not in_single_string:
                in_single_string = True
                result.append('"')
            else:
                in_single_string = False
                result.append('"')
            continue

        # Escape double quotes inside single-to-double converted strings
        if in_single_string and char == '"':
            result.append('\\"')
            continue

        result.append(char)

    return "".join(result)


def _fix_unescaped_newlines(text: str) -> str:
    """Replace literal newlines within JSON string values with \\n."""
    # Find string values and replace literal newlines within them
    result = []
    in_string = False
    escape_next = False

    for char in text:
        if escape_next:
            result.append(char)
            escape_next = False
            continue

        if char == "\\":
            result.append(char)
            escape_next = True
            continue

        if char == '"':
            in_string = not in_string
            result.append(char)
            continue

        if in_string and char == "\n":
            result.append("\\n")
            continue

        result.append(char)

    return "".join(result)


def _close_unclosed_brackets(text: str) -> str:
    """Append missing closing braces/brackets to make the JSON balanced."""
    stack: list[str] = []
    in_string = False
    escape_next = False

    for char in text:
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in ("}", "]"):
            if stack and stack[-1] == char:
                stack.pop()

    # Close any remaining open brackets
    if stack:
        text += "".join(reversed(stack))

    return text


def validate_against_schema(
    json_str: str,
    schema_keys: list[str],
) -> tuple[bool, list[str]]:
    """Validate a JSON string against expected top-level schema keys.

    Args:
        json_str: JSON string to validate.
        schema_keys: List of expected top-level keys.

    Returns:
        Tuple of (is_valid, list of error messages).
        is_valid is True if all expected keys are present and no unexpected keys exist.
    """
    errors: list[str] = []

    try:
        parsed = json.loads(json_str)
    except (json.JSONDecodeError, TypeError) as e:
        return (False, [f"Invalid JSON: {e}"])

    if not isinstance(parsed, dict):
        return (False, [f"Expected JSON object, got {type(parsed).__name__}"])

    expected = set(schema_keys)
    actual = set(parsed.keys())

    missing = expected - actual
    extra = actual - expected

    if missing:
        errors.append(f"Missing keys: {sorted(missing)}")
    if extra:
        errors.append(f"Unexpected keys: {sorted(extra)}")

    is_valid = len(missing) == 0 and len(extra) == 0
    return (is_valid, errors)


def postprocess_prediction(raw_output: str) -> dict[str, Any]:
    """Full postprocessing pipeline: extract -> fix -> parse -> validate.

    Args:
        raw_output: Raw model output text.

    Returns:
        Dict with keys:
            - "parsed": dict (the parsed JSON, or empty dict on failure)
            - "raw": str (the original raw output)
            - "valid": bool (whether the output is valid JSON)
            - "errors": list[str] (any errors encountered)
    """
    errors: list[str] = []

    # Step 1: Extract JSON from surrounding text
    extracted = extract_json(raw_output)

    # Step 2: Try parsing directly first
    parsed = _try_parse(extracted)

    if parsed is not None:
        return {
            "parsed": parsed,
            "raw": raw_output,
            "valid": True,
            "errors": [],
        }

    # Step 3: Apply fixes and retry
    fixed = fix_json(extracted)
    parsed = _try_parse(fixed)

    if parsed is not None:
        errors.append("JSON required fixing before parsing")
        return {
            "parsed": parsed,
            "raw": raw_output,
            "valid": True,
            "errors": errors,
        }

    # Step 4: Failed to parse even after fixing
    errors.append(f"Failed to parse JSON even after fixing. Extracted: {extracted[:200]}")
    return {
        "parsed": {},
        "raw": raw_output,
        "valid": False,
        "errors": errors,
    }


def _try_parse(text: str) -> dict[str, Any] | list | None:
    """Attempt to parse text as JSON, returning None on failure."""
    try:
        result = json.loads(text)
        if isinstance(result, (dict, list)):
            return result
        return None
    except (json.JSONDecodeError, TypeError):
        return None
