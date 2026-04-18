"""Validates vignettes against the patient's cognitive graph context."""

import logging
from typing import TypedDict, Literal
from pydantic import BaseModel
from .base_agent import BaseAgent
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
    forbidden = [(p, c) for (p, c), v in edges.items() if v == 0]

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
        edges = context.get("edges", {})
        self.system_prompt = _build_vignette_system_prompt(edges)
        user_prompt = VALIDATOR_VIGNETTE_USER_PROMPT.format(vignette=vignette)

        result: ValidationResult = self._invoke_structured(ValidationResult, self.system_prompt, user_prompt)
        if result is None:
            logger.warning("[%s] vignette validation parsing failed — defaulting to PASS", self.name)
            return {"passed": True, "violations": [], "feedback": None}

        self.log_response(user_prompt, result.model_dump_json(indent=2), output=result.model_dump())

        violations = self._filter_violations(result, edges)
        feedback   = self._build_feedback(violations)
        passed     = len(violations) == 0
        logger.info("[%s] %s", self.name, "PASS" if passed else f"FAIL — {len(violations)} violation(s)")
        return {"passed": passed, "violations": violations, "feedback": feedback}

    def _filter_violations(self, result: ValidationResult, edges: dict) -> list[dict]:
        """Return violations, stripping any edges the LLM incorrectly flagged as forbidden."""
        required = {f"{p} → {c}" for (p, c), v in edges.items() if v > 0}
        violations = [
            {"edge": v.edge, "explanation": v.explanation, "quote": v.quote}
            for v in result.violations if v.edge not in required
        ]
        removed = len(result.violations) - len(violations)
        if removed:
            flagged = [v.edge for v in result.violations if v.edge in required]
            logger.warning("[%s] stripped %d incorrectly flagged required edge(s): %s",
                           self.name, removed, flagged)
        return violations

    @staticmethod
    def _build_feedback(violations: list[dict]) -> str | None:
        """Format violations into a feedback string for the retry prompt, or None if none."""
        if not violations:
            return None
        lines = []
        for v in violations:
            lines.append(f"- {v['edge']}: {v['explanation']}")
            if v["quote"]:
                lines.append(f"  quote: \"{v['quote']}\"")
        return "\n".join(lines)

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
            if attempt < max_retries:
                vignette = retry_fn(result["feedback"])

        logger.warning("[%s] max retries reached — using last vignette", label)
        return vignette, attempts
