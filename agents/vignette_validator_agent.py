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
    reason:      str  # "Required" | "Required — Missing" | "Forbidden — Present"
    explanation: str  # one sentence describing the finding
    quote:       str  # direct quote from the vignette, or "" if edge is absent


class ValidationResult(BaseModel):
    verdict:         Literal["PASS", "FAIL"]
    violations:      list[EdgeViolation] = []  # missing required edges + present forbidden edges
    satisfied_edges: list[EdgeViolation] = []  # required edges correctly present, with supporting quote


class VignetteValidatorContext(TypedDict):
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

class VignetteValidatorAgent(BaseAgent):
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
            return {"passed": True, "violations": [], "satisfied_edges": [], "feedback": None}

        # Reclassify: "Forbidden — Present" entries wrongly placed in satisfied_edges → move to violations
        misplaced = [e for e in result.satisfied_edges if (e.reason or "").startswith("Forbidden")]
        if misplaced:
            logger.warning("[%s] reclassifying %d misplaced forbidden entry/entries from satisfied→violations",
                           self.name, len(misplaced))
            result.violations.extend(misplaced)
            result.satisfied_edges = [e for e in result.satisfied_edges if e not in misplaced]

        violations      = self._filter_violations(result, edges)
        satisfied_edges = [{"edge": e.edge, "reason": e.reason or "Required",
                            "explanation": e.explanation, "quote": e.quote}
                           for e in result.satisfied_edges]
        feedback        = self._build_feedback(violations)
        passed          = len(violations) == 0
        logger.info("[%s] %s", self.name, "PASS" if passed else f"FAIL — {len(violations)} violation(s)")
        return {"passed": passed, "violations": violations, "satisfied_edges": satisfied_edges, "feedback": feedback}

    # Phrases the LLM uses when a forbidden edge is correctly absent — not a real violation
    _ABSENT_PHRASES = ("does not", "no sentence", "not appear", "inactive", "is absent",
                       "cannot find", "not contain", "never appear")

    def _filter_violations(self, result: ValidationResult, edges: dict) -> list[dict]:
        """Strip two classes of LLM mistakes from the violations list:
        1. Required edges incorrectly filed as violations (should be in satisfied_edges).
        2. Forbidden edges that are correctly absent — the LLM filed them as violations
           but the explanation reveals no actual causal sentence was found.
        """
        required  = {f"{p} → {c}" for (p, c), v in edges.items() if v > 0}
        forbidden = {f"{p} → {c}" for (p, c), v in edges.items() if v == 0}
        satisfied_set = {e.edge for e in result.satisfied_edges}

        def _is_false_forbidden(v) -> bool:
            """True when LLM filed a forbidden edge but admits it isn't in the vignette."""
            if v.edge not in forbidden:
                return False
            expl = (v.explanation or "").lower()
            return any(phrase in expl for phrase in self._ABSENT_PHRASES)

        def _is_duplicate_required(v) -> bool:
            """True when LLM filed a required edge as violation AND as satisfied — keep the satisfied version."""
            return v.edge in required and v.edge in satisfied_set

        violations = [
            {"edge": v.edge, "reason": v.reason or ("Forbidden — Present" if v.edge in forbidden else "Required — Missing"),
             "explanation": v.explanation, "quote": v.quote}
            for v in result.violations
            if not _is_duplicate_required(v) and not _is_false_forbidden(v)
        ]
        removed = len(result.violations) - len(violations)
        if removed:
            flagged = [v.edge for v in result.violations
                       if _is_duplicate_required(v) or _is_false_forbidden(v)]
            logger.warning("[%s] stripped %d false violation(s): %s",
                           self.name, removed, flagged)
        return violations

    @staticmethod
    def _build_feedback(violations: list[dict]) -> str | None:
        """Format violations into a feedback string for the retry prompt, or None if none."""
        if not violations:
            return None
        missing  = [v for v in violations if "Missing"  in (v.get("reason") or "")]
        present  = [v for v in violations if "Present"  in (v.get("reason") or "")]
        lines = []
        if missing:
            lines.append("REQUIRED EDGES MISSING — add a sentence with explicit causal language for each:")
            for v in missing:
                lines.append(f"- {v['edge']}: {v['explanation']}")
        if present:
            lines.append("FORBIDDEN EDGES PRESENT — remove or rephrase explicit causal language for each:")
            for v in present:
                lines.append(f"- {v['edge']}: {v['explanation']}")
                if v["quote"]:
                    lines.append(f"  offending quote: \"{v['quote']}\"")
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
                "vignette":       vignette,
                "passed":         result["passed"],
                "violations":     result["violations"],
                "satisfied_edges": result["satisfied_edges"],
                "feedback":       result["feedback"],
            })

            if result["passed"]:
                logger.info("[%s] PASS on attempt %d", label, attempt + 1)
                return vignette, attempts

            logger.warning("[%s] FAIL on attempt %d", label, attempt + 1)
            if attempt < max_retries:
                vignette = retry_fn(result["feedback"])

        logger.warning("[%s] max retries reached — using last vignette", label)
        return vignette, attempts
