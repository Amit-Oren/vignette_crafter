"""
Pipeline step functions for vignette creation and validation.
Each step receives/returns a state dict.
"""
import logging
from pydantic import BaseModel
from typing import Literal
from agents.vignette_crafter_agent import VignetteCrafterAgent
from agents.zero_shot_vignette_agent import ZeroShotVignetteAgent
from agents.validator_agent import ValidatorAgent
from agents.persona_crafter_agent import PersonaCrafterAgent
from configs.prompts import VALIDATOR_INPUTS_PROMPT
from data.input.input import extract_aggregated_edges, get_demographics, get_self_report

logger = logging.getLogger(__name__)


class _InputValidationResult(BaseModel):
    verdict:   Literal["PASS", "FAIL"]
    reasoning: str


def _format_edges(agg_edges: dict) -> str:
    lines = [
        f"  {p} → {c}: {v:.2f}"
        for (p, c), v in agg_edges.items() if v > 0
    ]
    return "\n".join(lines) or "  (none)"


def _format_self_report(self_report: dict) -> str:
    lines = []
    for node, items in self_report.items():
        formatted = ", ".join(
            f"{i['key']}: {i['value']}" if isinstance(i, dict) else str(i)
            for i in items
        )
        lines.append(f"  {node}: {formatted}")
    return "\n".join(lines)


def _format_demographics(demo: dict) -> str:
    return "\n".join(f"  {k}: {v}" for k, v in demo.items())


def step_craft_persona(label: str, patient_id, llm, max_retries: int, n_items: int = 3) -> dict:
    """Sample demographics randomly, then use LLM (persona crafter) to select
    coherent self-report items. Validates the result and resamples demographics
    + re-selects on failure up to max_retries times.
    """
    agg_edges    = extract_aggregated_edges(patient_id)
    active_nodes = [node for (p, c), w in agg_edges.items() if w > 0 for node in (p, c)]
    active_nodes = list(dict.fromkeys(active_nodes))  # deduplicate, preserve order

    structured_llm = llm.with_structured_output(_InputValidationResult)
    attempts = []

    for attempt in range(max_retries + 1):
        # Step 1: sample random demographic anchors
        demographics = get_demographics(patient_id)

        # Step 2: PersonaCrafterAgent selects coherent self-report items
        persona_crafter = PersonaCrafterAgent(
            name=f"{label}_PersonaCrafter", role="PersonaCrafter", llm=llm,
            active_nodes=active_nodes, demographics=demographics, agg_edges=agg_edges,
            n_items=n_items,
        )
        self_report = persona_crafter.select()

        # Step 3: lightweight structural validation
        prompt = VALIDATOR_INPUTS_PROMPT.format(
            demographics=_format_demographics(demographics),
            edges=_format_edges(agg_edges),
            self_report=_format_self_report(self_report),
        )
        result: _InputValidationResult = structured_llm.invoke([
            {"role": "user", "content": prompt},
        ])

        passed = result.verdict == "PASS"
        attempts.append({"attempt": attempt + 1, "passed": passed, "reasoning": result.reasoning})
        logger.info("[%s] input validation attempt %d: %s — %s",
                    label, attempt + 1, result.verdict, result.reasoning)

        if passed:
            break
        if attempt < max_retries:
            logger.warning("[%s] re-sampling demographics and re-selecting self-report (attempt %d/%d)",
                           label, attempt + 1, max_retries)

    return {
        "agg_edges":              agg_edges,
        "demographics":           demographics,
        "self_report":            self_report,
        "input_validation_attempts": attempts,
    }


def step_persona(label: str, patient_id, llms: dict, persona_context: bool = False,
                 state: dict = None, use_formulation: bool = True) -> dict:
    # Reuse already-validated inputs from step_sample_inputs if available
    if state and "agg_edges" in state:
        agg_edges    = state["agg_edges"]
        demographics = state["demographics"]
        self_report  = state["self_report"]
    else:
        agg_edges    = extract_aggregated_edges(patient_id)
        demographics = get_demographics(patient_id)
        self_report  = get_self_report(patient_id)

    vignette_crafter = VignetteCrafterAgent(
        name=f"{label}_VignetteCrafter", role="VignetteCrafter",
        llm=llms["vignette_crafter"], patient_id=patient_id,
        use_demographics=persona_context, use_self_report=persona_context,
        use_formulation=use_formulation, demographics=demographics,
    )
    vignette = vignette_crafter.create_vignette()
    return {
        "vignette":          vignette,
        "vignette_attempts": [{"vignette": vignette, "passed": True, "feedback": None}],
        "agg_edges":         agg_edges,
        "demographics":      demographics,
        "self_report":       self_report,
        "_vignette_crafter": vignette_crafter,
    }


def step_zero_shot(label: str, patient_id, llms: dict) -> dict:
    agg_edges    = extract_aggregated_edges(patient_id)
    demographics = get_demographics(patient_id)
    self_report  = get_self_report(patient_id)

    agent = ZeroShotVignetteAgent(
        name=f"{label}_ZeroShot", role="ZeroShotVignetteCrafter",
        llm=llms["vignette_crafter"],
    )
    vignette = agent.create_vignette()
    return {
        "vignette":          vignette,
        "vignette_attempts": [{"vignette": vignette, "passed": True, "feedback": None}],
        "agg_edges":         agg_edges,
        "demographics":      demographics,
        "self_report":       self_report,
    }


def step_validate_persona(label: str, state: dict, validator: ValidatorAgent, max_retries: int) -> dict:
    edges = state["agg_edges"]
    strong   = sum(1 for v in edges.values() if v >= 0.7)
    moderate = sum(1 for v in edges.values() if 0.4 <= v < 0.7)
    forbidden = sum(1 for v in edges.values() if v == 0.0)
    logger.info("[%s] Validator — context: %d strong, %d moderate, %d forbidden edges",
                label, strong, moderate, forbidden)

    vignette, attempts = validator.validate_with_retry(
        initial_vignette=state["vignette"],
        context={"edges": edges},
        retry_fn=state["_vignette_crafter"].create_vignette_with_feedback,
        max_retries=max_retries,
        label=label,
    )
    return {"vignette": vignette, "vignette_attempts": attempts}
