"""
Offline eval runner: batch evaluation against a golden dataset.

Runs a full eval suite (retrieval + generation + LLM judge) over a JSONL
golden set and produces a structured report. Designed to run in CI on every
PR that touches src/ or prompts/.

Report format: EvalReport with per-metric scores, regression flags, and
a markdown summary suitable for PR comments.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EvalCase:
    id: str
    question: str
    expected_chunk_ids: list[str]
    expected_answer_contains: list[str]
    category: str
    reference_answer: str = ""


@dataclass
class CaseResult:
    case_id: str
    question: str
    retrieved_chunk_ids: list[str]
    answer: str
    recall_at_5: float
    mrr: float
    faithfulness: float
    relevance: float
    latency_ms: float
    error: str | None = None


@dataclass
class EvalReport:
    run_id: str
    timestamp: str
    n_cases: int
    recall_at_5: float
    mrr: float
    faithfulness: float
    relevance: float
    avg_latency_ms: float
    case_results: list[CaseResult] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)

    def to_pr_comment(self, baseline: "EvalReport | None" = None) -> str:
        lines = [
            "## Eval Results",
            "",
            f"**Run:** `{self.run_id}` | **Cases:** {self.n_cases} | **Avg latency:** {self.avg_latency_ms:.0f}ms",
            "",
            "| Metric | This PR |" + (" Baseline | Delta |" if baseline else ""),
            "|--------|---------|" + ("----------|-------|" if baseline else ""),
        ]

        metrics = [
            ("Recall@5", self.recall_at_5, baseline.recall_at_5 if baseline else None),
            ("MRR", self.mrr, baseline.mrr if baseline else None),
            ("Faithfulness", self.faithfulness, baseline.faithfulness if baseline else None),
            ("Relevance", self.relevance, baseline.relevance if baseline else None),
        ]

        for name, value, base_value in metrics:
            if baseline and base_value is not None:
                delta = value - base_value
                sign = "+" if delta >= 0 else ""
                flag = " ⚠️" if delta < -0.02 else (" ✅" if delta > 0.01 else "")
                lines.append(
                    f"| {name} | {value:.3f} | {base_value:.3f} | {sign}{delta:.3f}{flag} |"
                )
            else:
                lines.append(f"| {name} | {value:.3f} |")

        if self.regressions:
            lines.extend(["", "### Regressions", ""])
            for r in self.regressions:
                lines.append(f"- {r}")

        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps({
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "n_cases": self.n_cases,
            "recall_at_5": self.recall_at_5,
            "mrr": self.mrr,
            "faithfulness": self.faithfulness,
            "relevance": self.relevance,
            "avg_latency_ms": self.avg_latency_ms,
            "regressions": self.regressions,
        }, indent=2)


REGRESSION_THRESHOLDS = {
    "recall_at_5": 0.02,
    "mrr": 0.02,
    "faithfulness": 0.03,
    "relevance": 0.03,
}


class OfflineEvalRunner:
    """
    Runs offline evaluation against a golden JSONL dataset.

    Each golden case specifies a question, expected chunk IDs (for retrieval
    metrics), and expected answer keywords (for sanity checking).

    The runner calls the full RAG pipeline for each case, then scores:
    - Retrieval: Recall@5, MRR (comparing retrieved IDs to expected IDs)
    - Generation: Faithfulness and Relevance (via LLM judge)

    Regression detection: if any metric drops > threshold vs baseline,
    the run is flagged as a regression and CI exits with code 1.
    """

    def __init__(
        self,
        golden_path: str,
        rag_pipeline=None,
        judge=None,
        baseline_report_path: str | None = None,
        concurrency: int = 5,
    ):
        self.golden_path = Path(golden_path)
        self.rag_pipeline = rag_pipeline
        self.judge = judge
        self.baseline_report_path = Path(baseline_report_path) if baseline_report_path else None
        self.concurrency = concurrency

    def load_golden(self) -> list[EvalCase]:
        cases = []
        for line in self.golden_path.read_text().strip().splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            cases.append(EvalCase(
                id=data["id"],
                question=data["question"],
                expected_chunk_ids=data.get("expected_chunk_ids", []),
                expected_answer_contains=data.get("expected_answer_contains", []),
                category=data.get("category", "general"),
                reference_answer=data.get("reference_answer", ""),
            ))
        return cases

    async def run(self) -> EvalReport:
        """Run eval on all golden cases and return a report."""
        import uuid
        from datetime import datetime

        cases = self.load_golden()
        logger.info(f"Running eval on {len(cases)} cases (concurrency={self.concurrency})")

        semaphore = asyncio.Semaphore(self.concurrency)
        tasks = [self._eval_case(case, semaphore) for case in cases]
        case_results: list[CaseResult] = await asyncio.gather(*tasks)

        valid = [r for r in case_results if r.error is None]

        recall = sum(r.recall_at_5 for r in valid) / len(valid) if valid else 0.0
        mrr = sum(r.mrr for r in valid) / len(valid) if valid else 0.0
        faithfulness = sum(r.faithfulness for r in valid) / len(valid) if valid else 0.0
        relevance = sum(r.relevance for r in valid) / len(valid) if valid else 0.0
        avg_latency = sum(r.latency_ms for r in valid) / len(valid) if valid else 0.0

        run_id = str(uuid.uuid4())[:8]
        report = EvalReport(
            run_id=run_id,
            timestamp=datetime.utcnow().isoformat(),
            n_cases=len(cases),
            recall_at_5=recall,
            mrr=mrr,
            faithfulness=faithfulness,
            relevance=relevance,
            avg_latency_ms=avg_latency,
            case_results=case_results,
        )

        report.regressions = self._detect_regressions(report)
        return report

    async def _eval_case(self, case: EvalCase, semaphore: asyncio.Semaphore) -> CaseResult:
        async with semaphore:
            t0 = time.time()
            try:
                chunks, answer = await self._run_pipeline(case.question)
                latency_ms = (time.time() - t0) * 1000

                retrieved_ids = [c.get("chunk_id", "") for c in chunks]

                from src.metrics.retrieval import recall_at_k, mean_reciprocal_rank
                recall = recall_at_k(retrieved_ids, case.expected_chunk_ids, k=5)
                mrr = mean_reciprocal_rank(retrieved_ids, case.expected_chunk_ids)

                faithfulness, relevance = 0.0, 0.0
                if self.judge and chunks:
                    result = await self.judge.score(
                        question=case.question,
                        context_chunks=[c.get("text", "") for c in chunks],
                        answer=answer,
                    )
                    faithfulness = result.faithfulness
                    relevance = result.relevance

                return CaseResult(
                    case_id=case.id,
                    question=case.question,
                    retrieved_chunk_ids=retrieved_ids,
                    answer=answer,
                    recall_at_5=recall,
                    mrr=mrr,
                    faithfulness=faithfulness,
                    relevance=relevance,
                    latency_ms=latency_ms,
                )

            except Exception as e:
                logger.error(f"Case {case.id} failed: {e}")
                return CaseResult(
                    case_id=case.id,
                    question=case.question,
                    retrieved_chunk_ids=[],
                    answer="",
                    recall_at_5=0.0,
                    mrr=0.0,
                    faithfulness=0.0,
                    relevance=0.0,
                    latency_ms=(time.time() - t0) * 1000,
                    error=str(e),
                )

    async def _run_pipeline(self, question: str) -> tuple[list[dict], str]:
        """Run the RAG pipeline. Override for different pipeline implementations."""
        if self.rag_pipeline is None:
            raise NotImplementedError("Provide a rag_pipeline or override _run_pipeline()")
        return await self.rag_pipeline.query(question)

    def _detect_regressions(self, report: EvalReport) -> list[str]:
        if not self.baseline_report_path or not self.baseline_report_path.exists():
            return []

        baseline = json.loads(self.baseline_report_path.read_text())
        regressions = []

        checks = [
            ("recall_at_5", report.recall_at_5, baseline.get("recall_at_5", 0)),
            ("mrr", report.mrr, baseline.get("mrr", 0)),
            ("faithfulness", report.faithfulness, baseline.get("faithfulness", 0)),
            ("relevance", report.relevance, baseline.get("relevance", 0)),
        ]

        for metric, current, base in checks:
            threshold = REGRESSION_THRESHOLDS.get(metric, 0.02)
            delta = current - base
            if delta < -threshold:
                regressions.append(
                    f"{metric}: {current:.3f} vs baseline {base:.3f} (Δ={delta:+.3f}, threshold={-threshold})"
                )

        return regressions
