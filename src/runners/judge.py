"""
LLM-as-judge for faithfulness and relevance scoring.

Judge model: claude-3-5-sonnet-20241022 (default).
Prompts are versioned in src/runners/prompts/ and pinned by content hash.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anthropic


JUDGE_MODEL = "claude-3-5-sonnet-20241022"
PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> tuple[str, str]:
    """Load prompt template and return (template, content_hash)."""
    path = PROMPTS_DIR / f"{name}.txt"
    template = path.read_text()
    content_hash = hashlib.sha256(template.encode()).hexdigest()[:8]
    return template, content_hash


class Judge:
    """
    LLM-as-judge using structured output for consistent scoring.

    Scores are 0.0-1.0. Each call returns a score and a brief reasoning string.
    Reasoning enables debugging: "why did this response score 0.3 on faithfulness?"
    """

    def __init__(self, model: str = JUDGE_MODEL, max_tokens: int = 256):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self._faithfulness_prompt, self._faithfulness_hash = _load_prompt("faithfulness")
        self._relevance_prompt, self._relevance_hash = _load_prompt("relevance")

    async def score_faithfulness(
        self, query: str, answer: str, reference: str
    ) -> tuple[float, str]:
        """
        Score how faithful the answer is to the reference.

        Faithfulness: does the answer contain only information present in the reference?
        A score of 1.0 means every claim in the answer is grounded in the reference.
        A score of 0.0 means the answer contradicts or ignores the reference.
        """
        prompt = self._faithfulness_prompt.format(
            query=query, answer=answer, reference=reference
        )
        return await self._score(prompt)

    async def score_relevance(self, query: str, answer: str) -> tuple[float, str]:
        """
        Score how relevant the answer is to the query.

        Relevance: does the answer address what was asked?
        """
        prompt = self._relevance_prompt.format(query=query, answer=answer)
        return await self._score(prompt)

    async def _score(self, prompt: str) -> tuple[float, str]:
        """Call the judge model and parse the structured response."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            tools=[{
                "name": "submit_score",
                "description": "Submit the evaluation score",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "score": {
                            "type": "number",
                            "description": "Score between 0.0 and 1.0",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "One sentence explaining the score",
                        },
                    },
                    "required": ["score", "reasoning"],
                },
            }],
            tool_choice={"type": "tool", "name": "submit_score"},
        )

        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_score":
                result = block.input
                return float(result["score"]), result["reasoning"]

        return 0.0, "Judge failed to produce a score"

    @property
    def prompt_versions(self) -> dict:
        return {
            "faithfulness": self._faithfulness_hash,
            "relevance": self._relevance_hash,
        }
