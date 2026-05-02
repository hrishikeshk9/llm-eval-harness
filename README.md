# llm-eval-harness

A practical evaluation framework for LLM systems. Offline golden-set regression, online production sampling, LLM-as-judge scoring, and a CI gate that posts results as PR comments.

Eval is the #1 differentiator between LLM systems that ship confidently and ones that regress silently. This harness makes eval a first-class part of the development loop — not a post-hoc audit.

---

## What This Solves

| Without eval harness | With eval harness |
|---|---|
| "Looks good to me" before merging | CI gate blocks merge on metric regression |
| Prompt changes break quality silently | Golden-set regression detected before prod |
| No idea if new model version is better | Automated comparison with structured output |
| Production quality unknown until user complaints | Online sampling + weekly drift report |

---

## Architecture

```mermaid
flowchart LR
    subgraph Offline
        GoldenSet[(Golden Set\nJSONL)] --> OfflineRunner[Offline Runner]
        OfflineRunner --> RetrievalMetrics[Recall@k · MRR · nDCG]
        OfflineRunner --> GenMetrics[Faithfulness · Relevance]
        GenMetrics --> Judge[LLM Judge\nclaude-3-5-sonnet]
        RetrievalMetrics --> Report[Eval Report]
        Judge --> Report
        Report --> CIGate{CI Gate\npass / fail}
        Report --> PRComment[PR Comment]
    end

    subgraph Online
        ProdTraffic[2% Production\nSampling] --> OnlineRunner[Online Runner]
        OnlineRunner --> RefFreeMetrics[Reference-free\nmetrics]
        RefFreeMetrics --> Dashboard[Grafana Dashboard]
        RefFreeMetrics --> SlackReport[Weekly Slack Report]
    end
```

---

## Quickstart

```bash
pip install -e ".[dev]"

# Run offline eval on a golden set
python -m src.runners.offline \
  --golden-set tests/golden/qa_golden.jsonl \
  --baseline tests/golden/baseline.json \
  --output /tmp/report.md

# CI: exits 1 if regressions detected
echo "Exit code: $?"
```

---

## Metrics

### Retrieval Metrics (`src/metrics/retrieval.py`)

| Metric | What it measures |
|---|---|
| Recall@k | Fraction of relevant chunks in top-k |
| MRR | Mean Reciprocal Rank of first relevant chunk |
| nDCG@k | Normalized Discounted Cumulative Gain |
| Precision@k | Fraction of top-k chunks that are relevant |

### Generation Metrics (`src/metrics/generation.py`)

| Metric | Method |
|---|---|
| Faithfulness | LLM judge: does the answer stay within the source? |
| Relevance | LLM judge: does the answer address the query? |
| Completeness | LLM judge: does the answer cover the key points? |
| Citation accuracy | Structural: source chunk ID grounding check |

### Safety Metrics (`src/metrics/safety.py`)

| Metric | Method |
|---|---|
| PII leakage | Regex + NER scan of generated text |
| Jailbreak success rate | Adversarial prompt set pass rate |
| Topic drift | Embedding distance from allowed topic space |

---

## Golden Set Format

```jsonl
{"query": "What is the data retention policy?", "expected_chunk_ids": ["doc_123_chunk_4", "doc_123_chunk_5"], "reference_answer": "Data is retained for 7 years per regulatory requirement.", "metadata": {"category": "compliance", "difficulty": "easy"}}
{"query": "Summarize the Q3 financial results", "expected_chunk_ids": ["fin_q3_chunk_1"], "reference_answer": "Revenue grew 12% YoY...", "metadata": {"category": "finance", "difficulty": "hard"}}
```

---

## CI Integration

GitHub Actions workflow included: [.github/workflows/eval.yml](.github/workflows/eval.yml)

```yaml
# Triggers on PR to main when src/ or tests/golden/ changes
# Posts eval table as PR comment
# Exits 1 if Recall@5 drops >3% or Faithfulness drops >5%
```

CI failure example:

```
## RAG Eval Results

| Metric | Score | Baseline | Delta |
|---|---|---|---|
| Recall@5 ⚠️ | 0.71 | 0.79 | -0.08 |
| MRR | 0.73 | 0.74 | -0.01 |
| Faithfulness | 0.85 | 0.86 | -0.01 |
| Relevance | 0.82 | 0.81 | +0.01 |

**CI gate failed.** Regressions detected: Recall@5
```

---

## Judge Model

Default judge: `claude-3-5-sonnet-20241022`. Prompts are versioned in `src/runners/prompts/` and pinned by content hash in `config/judge_config.yaml`.

Changing a judge prompt requires:
1. Update the prompt file
2. Run `python -m src.runners.judge_calibration` (compares new scores against 50 human-labeled examples)
3. Calibration must show r≥0.75 Spearman on faithfulness before the new prompt can be merged

This prevents prompt changes from silently shifting your eval baseline.

---

## Repository Structure

```
.
├── src/
│   ├── runners/
│   │   ├── offline.py          # Golden set regression runner
│   │   ├── online.py           # Production sampling runner
│   │   ├── judge.py            # LLM-as-judge
│   │   └── prompts/            # Versioned judge prompts
│   ├── metrics/
│   │   ├── retrieval.py        # Recall@k, MRR, nDCG
│   │   ├── generation.py       # Faithfulness, relevance, completeness
│   │   └── safety.py           # PII, jailbreak, topic drift
│   └── datasets/
│       └── golden_sets/        # Versioned golden sets (JSONL)
├── config/
│   └── judge_config.yaml       # Judge model + prompt hash pins
├── examples/
│   ├── ci-gate/                # GitHub Actions workflow
│   └── nightly-regression/     # Scheduled eval example
├── tests/
│   └── golden/                 # Sample golden sets
└── .github/workflows/
    └── eval.yml
```

---

## Related

- [rag-reference-architecture](https://github.com/hrishikeshk9/rag-reference-architecture) — uses this harness for its CI eval gate
- [architecture-decision-records/rag/0003-eval-strategy](https://github.com/hrishikeshk9/architecture-decision-records/blob/main/rag/0003-eval-strategy.md) — decision record explaining why this eval approach was chosen

---

## License

Apache-2.0
