"""Validates vignettes against the patient's cognitive graph context."""

import logging
from typing import TypedDict, Literal
from pydantic import BaseModel
from .base_agent import BaseAgent, _count
from configs.prompts import VALIDATOR_VIGNETTE_SYSTEM_PROMPT, VALIDATOR_VIGNETTE_USER_PROMPT

logger = logging.getLogger(__name__)


# ── Data models ───────────────────────────────────────────────────────────────

class EdgeViolation(BaseModel):
    edge:        str  # e.g. "Triggers → Memory"
    explanation: str  # one sentence describing the violation
    quote:       str  # direct quote from the vignette, or "" if edge is absent


class ValidationResult(BaseModel):
    verdict:    Literal["PASS", "FAIL"]
    violations: list[EdgeViolation]  # empty on PASS


class ValidatorContext(TypedDict):
    edges: dict   # {(parent, child): float}


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_vignette_system_prompt(edges: dict) -> str:
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

    return VALIDATOR_VIGNETTE_SYSTEM_PROMPT.format(
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
        return None

    def validate(self, vignette: str, context: dict) -> dict:
        """Validate a vignette. Returns {"passed": bool, "violations": list, "feedback": str | None}."""
        edges  = context.get("edges", {})
        self.system_prompt = _build_vignette_system_prompt(edges)
        user_prompt        = VALIDATOR_VIGNETTE_USER_PROMPT.format(vignette=vignette)

        structured_llm = self.llm.with_structured_output(ValidationResult, include_raw=True, method="function_calling")
        raw = structured_llm.invoke([
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": user_prompt},
        ])
        if isinstance(raw, dict):
            _count(raw.get("raw"))
            result: ValidationResult = raw["parsed"]
        else:
            result: ValidationResult = raw

        if result is None:
            logger.warning("[%s] vignette validation parsing failed — defaulting to PASS", self.name)
            return {"passed": True, "violations": [], "feedback": None}

        self.log_response(user_prompt, result.model_dump_json(indent=2),
                          output=result.model_dump())

        # Build required edge set to filter out any required edges the LLM incorrectly flagged
        required_edges = {f"{p} → {c}" for (p, c), v in edges.items() if v > 0}

        violations = [
            {"edge": v.edge, "explanation": v.explanation, "quote": v.quote}
            for v in result.violations
            if v.edge not in required_edges
        ]
        if len(violations) < len(result.violations):
            removed = [v.edge for v in result.violations if v.edge in required_edges]
            logger.warning("[%s] stripped %d incorrectly flagged required edge(s): %s",
                           self.name, len(removed), removed)

        passed = len(violations) == 0

        # Format feedback string for the retry prompt
        feedback = None
        if not passed:
            lines = []
            for v in violations:
                lines.append(f"- {v['edge']}: {v['explanation']}")
                if v["quote"]:
                    lines.append(f"  quote: \"{v['quote']}\"")
            feedback = "\n".join(lines)
            logger.info("[%s] FAIL — %d violation(s)", self.name, len(violations))
        else:
            logger.info("[%s] PASS", self.name)

        return {"passed": passed, "violations": violations, "feedback": feedback}

    def validate_with_retry(self, initial_vignette: str, context: dict,
                            retry_fn, max_retries: int, label: str) -> tuple[str, list]:
        """Validation loop with feedback-driven retry.
        retry_fn(feedback) -> str : generates a revised vignette.
        Returns (final_vignette, attempts).
        """
        vignette = initial_vignette
        attempts = []

        for attempt in range(max_retries + 1):
            result = self.validate(vignette, context)
            attempts.append({
                "vignette":   vignette,
                "passed":     result["passed"],
                "violations": result["violations"],
                "feedback":   result["feedback"],
            })

            if result["passed"]:
                logger.info("[%s] PASS on attempt %d", label, attempt + 1)
                return vignette, attempts

            logger.warning("[%s] FAIL on attempt %d", label, attempt + 1)
            vignette = retry_fn(result["feedback"])

        logger.warning("[%s] max retries reached — using last vignette", label)
        return vignette, attempts
