"""Validates persona profiles for clinical plausibility in two stages:
1. Demographics-only consistency (before self-report selection)
2. Self-report alignment with demographics and trauma type (after selection)
"""

import logging
from typing import Literal
from pydantic import BaseModel
from .base_agent import BaseAgent, _count
from configs.prompts import (
    PERSONA_VALIDATOR_DEMOGRAPHICS_SYSTEM_PROMPT,
    PERSONA_VALIDATOR_DEMOGRAPHICS_USER_PROMPT,
    PERSONA_VALIDATOR_SELFREPORT_SYSTEM_PROMPT,
    PERSONA_VALIDATOR_SELFREPORT_USER_PROMPT,
)

logger = logging.getLogger(__name__)


class FieldIssue(BaseModel):
    field:       str  # one of: age, gender, nationality, relationship_status, trauma_type, pcl5
    explanation: str  # one sentence explaining why it was flagged


class DemographicsValidationResult(BaseModel):
    verdict: Literal["PASS", "FAIL"]
    issues:  list[FieldIssue]  # empty on PASS


class SelfReportItemIssue(BaseModel):
    component:   str  # node name e.g. "Triggers", "Memory"
    item:        str  # the item key
    explanation: str  # one sentence explaining why it was flagged


class SelfReportValidationResult(BaseModel):
    verdict: Literal["PASS", "FAIL"]
    issues:  list[SelfReportItemIssue]  # empty on PASS


class PersonaValidatorAgent(BaseAgent):
    def __init__(self, name: str, role: str, llm):
        super().__init__(name, role, "", llm)

    def setup_agent(self):
        return None

    def validate_demographics(self, demographics: dict) -> dict:
        """Check internal consistency of demographics (age/relationship/trauma type).
        Returns {"passed": bool, "issues": [{"field": str, "explanation": str}], "problematic_fields": [str]}.
        """
        self.system_prompt = PERSONA_VALIDATOR_DEMOGRAPHICS_SYSTEM_PROMPT
        user_prompt = PERSONA_VALIDATOR_DEMOGRAPHICS_USER_PROMPT.format(
            demographics=self.fmt_demographics(demographics),
        )
        structured_llm = self.llm.with_structured_output(DemographicsValidationResult, include_raw=True, method="function_calling")
        raw = structured_llm.invoke([
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": user_prompt},
        ])
        if isinstance(raw, dict):
            _count(raw.get("raw"))
            result: DemographicsValidationResult = raw["parsed"]
        else:
            result: DemographicsValidationResult = raw

        if result is None:
            logger.warning("[%s] demographics validation parsing failed — defaulting to PASS", self.name)
            return {"passed": True, "issues": [], "problematic_fields": []}

        self.log_response(user_prompt, result.model_dump_json(indent=2),
                          output=result.model_dump())

        passed = result.verdict == "PASS"
        issues = [{"field": i.field, "explanation": i.explanation} for i in result.issues]
        problematic_fields = [i.field for i in result.issues]
        logger.info("[%s] demographics validation: %s — %s", self.name, result.verdict,
                    ", ".join(f"{i['field']}: {i['explanation']}" for i in issues) or "no issues")
        return {"passed": passed, "issues": issues, "problematic_fields": problematic_fields}

    def validate_self_report(self, demographics: dict, self_report: dict) -> dict:
        """Check that self-report items are coherent with demographics and trauma type.
        Returns {"passed": bool, "issues": [...], "problematic_items": {node: [bad_keys]}}.
        """
        self.system_prompt = PERSONA_VALIDATOR_SELFREPORT_SYSTEM_PROMPT
        user_prompt = PERSONA_VALIDATOR_SELFREPORT_USER_PROMPT.format(
            demographics=self.fmt_demographics(demographics),
            self_report=self.fmt_self_report(self_report),
        )
        structured_llm = self.llm.with_structured_output(SelfReportValidationResult, include_raw=True, method="function_calling")
        raw = structured_llm.invoke([
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": user_prompt},
        ])
        if isinstance(raw, dict):
            _count(raw.get("raw"))
            result: SelfReportValidationResult = raw["parsed"]
        else:
            result: SelfReportValidationResult = raw

        if result is None:
            logger.warning("[%s] self-report validation parsing failed — defaulting to PASS", self.name)
            return {"passed": True, "issues": [], "problematic_items": {}}

        self.log_response(user_prompt, result.model_dump_json(indent=2),
                          output=result.model_dump())

        passed = result.verdict == "PASS"
        issues = [{"component": i.component, "item": i.item, "explanation": i.explanation}
                  for i in result.issues]
        # build problematic_items dict for the fix step
        # Normalize: LLM sometimes returns "key: value" — extract just the key part
        problematic_items = {}
        for i in result.issues:
            key = i.item.split(":")[0].strip()
            problematic_items.setdefault(i.component, []).append(key)
        logger.info("[%s] self-report validation: %s — %s", self.name, result.verdict,
                    ", ".join(f"{i['component']}/{i['item']}: {i['explanation']}" for i in issues) or "no issues")
        return {"passed": passed, "issues": issues, "problematic_items": problematic_items}

