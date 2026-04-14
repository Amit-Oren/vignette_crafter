"""Validates vignettes against the patient's cognitive graph context."""

import logging
from typing import TypedDict, Literal
from pydantic import BaseModel
from .base_agent import BaseAgent, _count
from configs.prompts import VALIDATOR_VIGNETTE_PROMPT

logger = logging.getLogger(__name__)


# ── Data models ───────────────────────────────────────────────────────────────

class ValidationResult(BaseModel):
    verdict:   Literal["PASS", "FAIL"]
    reasoning: str


class ValidatorContext(TypedDict):
    edges: dict   # {(parent, child): float}


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(edges: dict) -> str:
    active = set()
    inactive_candidates = set()
    for (p, c), val in edges.items():
        if val > 0:
            active.update([p, c])
        else:
            inactive_candidates.update([p, c])
    inactive  = inactive_candidates - active
    required  = [(p, c) for (p, c), v in edges.items() if v > 0]
    forbidden = [(p, c) for (p, c), v in edges.items() if v == 0.0]

    return VALIDATOR_VIGNETTE_PROMPT.format(
        active_components   = "\n".join(f"  - {n}" for n in sorted(active))   or "  (none)",
        inactive_components = "\n".join(f"  - {n}" for n in sorted(inactive)) or "  (none)",
        required_edges      = "\n".join(f"  - {p} → {c}" for p, c in required)  or "  (none)",
        forbidden_edges     = "\n".join(f"  - {p} → {c}" for p, c in forbidden) or "  (none)",
    )


# ── Agent ─────────────────────────────────────────────────────────────────────

class ValidatorAgent(BaseAgent):
    def __init__(self, name: str, role: str, llm):
        super().__init__(name, role, "", llm)

    def setup_agent(self):
        return None   # validator uses structured LLM directly, not a conversational agent

    def validate(self, vignette: str, context: dict) -> dict:
        """Validate a vignette. Returns {"passed": bool, "feedback": str | None}."""
        edges  = context.get("edges", {})
        prompt = _build_prompt(edges)

        structured_llm = self.llm.with_structured_output(ValidationResult, include_raw=True)
        raw = structured_llm.invoke([
            {"role": "system", "content": prompt},
            {"role": "user",   "content": f"Vignette:\n{vignette}"},
        ])
        if isinstance(raw, dict):   # native OpenAI: {"raw": AIMessage, "parsed": ...}
            _count(raw.get("raw"))
            result: ValidationResult = raw["parsed"]
        else:                        # custom wrappers (DeepSeek/OpenSource) return model directly
            result: ValidationResult = raw
        self.log_response(vignette, result.model_dump_json(indent=2))

        passed = result.verdict == "PASS"
        logger.info("[%s] verdict: %s", self.name, result.verdict)
        if not passed:
            logger.info("[%s] reason: %s", self.name, result.reasoning)

        return {"passed": passed, "feedback": None if passed else result.reasoning}

    def validate_with_retry(self, initial_vignette: str, context: dict,
                            retry_fn, max_retries: int, label: str) -> tuple[str, list]:
        """Validation loop with feedback-driven retry.
        retry_fn(feedback) -> str : generates a revised vignette.
        Returns (final_vignette, attempts).
        """
        vignette = initial_vignette
        attempts = []

        for attempt in range(max_retries):
            result = self.validate(vignette, context)
            attempts.append({"vignette": vignette, "passed": result["passed"], "feedback": result["feedback"]})

            if result["passed"]:
                logger.info("[%s] PASS on attempt %d", label, attempt + 1)
                return vignette, attempts

            logger.warning("[%s] FAIL on attempt %d: %s", label, attempt + 1, result["feedback"])
            vignette = retry_fn(result["feedback"])

        logger.warning("[%s] max retries reached — using last vignette", label)
        return vignette, attempts
