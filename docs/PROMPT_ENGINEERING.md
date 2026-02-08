# Prompt Engineering Methodology

## Overview

Systematic evaluation of 7 prompt strategies for document information extraction using Qwen2-VL-2B-Instruct, organized in three categories with increasing sophistication.

## Strategy Categories

### Zero-Shot Strategies

**1. Zero-Shot Basic**
- Minimal instruction: "Extract all fields from this receipt image as JSON"
- Tests the model's default document understanding ability
- Baseline for comparison

**2. Zero-Shot Detailed**
- Includes role assignment ("document extraction specialist")
- Specifies categories (menu, total, sub_total, etc.)
- More directive about output format

**3. Zero-Shot Structured**
- Provides explicit JSON schema with field descriptions
- Includes formatting rules (null for missing fields, preserve text exactly)
- Most constrained zero-shot approach

### Few-Shot Strategies

**4. Few-Shot (2 examples)**
- Two representative receipt extractions as examples
- Selected for diversity (simple receipt + complex receipt)
- Tests in-context learning with minimal examples

**5. Few-Shot (5 examples)**
- Five examples covering edge cases
- Includes receipts with: many items, discounts, multiple payment methods, sparse fields
- Tests whether more examples improve extraction

### Chain-of-Thought Strategies

**6. CoT Step-by-Step**
- Explicit reasoning steps: identify layout → read items → find totals → compile JSON
- Model must think through extraction process before outputting JSON
- Response after "RESULT:" marker

**7. CoT Self-Verification**
- Extract first, then verify with arithmetic checks
- Checks: item prices sum to subtotal, subtotal + tax = total
- Self-correction mechanism

## Evaluation Protocol

- **Test set**: 50 samples from CORD v2 test split
- **Metrics**: Field F1 (micro/macro), JSON validity, schema compliance, ANLS
- **Controls**: Same model, same samples, same generation parameters across all strategies
- **Generation config**: temperature=0.1, max_new_tokens=1024, do_sample=False

## Design Decisions

### Why These 7 Strategies?

The strategies form a progression that isolates the effect of different prompt engineering techniques:
- **Basic → Detailed → Structured**: Measures impact of instruction specificity
- **Zero-shot → Few-shot**: Measures impact of in-context examples
- **Direct → CoT**: Measures impact of reasoning before answering
- **CoT → Self-Verify**: Measures impact of self-correction

### Example Selection for Few-Shot

Examples selected based on:
1. **Representativeness**: Cover common receipt layouts
2. **Diversity**: Different numbers of items, payment types, languages
3. **Quality**: Clean, unambiguous ground truth
4. **Complexity gradient**: Simple to complex

### Schema Design

The output schema mirrors CORD v2's annotation structure:
- Hierarchical JSON with superclass grouping
- Lists for repeating elements (menu items)
- Null for absent fields (not empty strings)

## Expected Findings

Based on prior work in document extraction with VLMs:

1. **Structured zero-shot** likely outperforms basic — schema guidance significantly helps
2. **Few-shot** should improve format consistency but may hit context length limits
3. **CoT** may help with complex receipts but add noise for simple ones
4. **Self-verify** useful for arithmetic consistency but adds latency
5. **Best strategy** will serve as the "prompted" baseline for fine-tuning comparison
