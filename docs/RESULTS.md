# DocuMind Results

## Performance Comparison

> **Note**: Results below are placeholder templates. Run the full pipeline to populate with actual numbers.

### Overall Metrics

| Metric | Base Model | Best Prompt | Fine-Tuned | Δ (Base→FT) |
|--------|-----------|-------------|------------|--------------|
| Field F1 (micro) | — | — | — | — |
| Field F1 (macro) | — | — | — | — |
| Exact Match | — | — | — | — |
| ANLS | — | — | — | — |
| JSON Validity | — | — | — | — |
| Schema Compliance | — | — | — | — |

### LLM Judge Scores (1-5 scale)

| Dimension | Base | Prompted | Fine-Tuned |
|-----------|------|----------|------------|
| Completeness | — | — | — |
| Accuracy | — | — | — |
| Format | — | — | — |
| **Average** | — | — | — |

### Prompt Strategy Comparison

| Strategy | Field F1 | JSON Validity | Schema Compliance |
|----------|----------|---------------|-------------------|
| Zero-Shot Basic | — | — | — |
| Zero-Shot Detailed | — | — | — |
| Zero-Shot Structured | — | — | — |
| Few-Shot (2) | — | — | — |
| Few-Shot (5) | — | — | — |
| CoT Step-by-Step | — | — | — |
| CoT Self-Verify | — | — | — |

### Per-Field Performance (Fine-Tuned)

| Field | Precision | Recall | F1 |
|-------|-----------|--------|----|
| menu.nm | — | — | — |
| menu.price | — | — | — |
| total.total_price | — | — | — |
| sub_total.subtotal_price | — | — | — |
| ... | ... | ... | ... |

## Key Findings

1. **Prompt engineering impact**: [To be filled after experiments]
2. **Fine-tuning gains**: [To be filled after training]
3. **Per-field analysis**: [To be filled after evaluation]
4. **Cross-domain generalization**: [FUNSD results to be added]

## Reproducing Results

```bash
# Full pipeline
make pipeline

# Or step by step
make data
make prompts
make train
make evaluate
```

## W&B Dashboard

Training metrics and comparison charts available at: [W&B project link to be added]
