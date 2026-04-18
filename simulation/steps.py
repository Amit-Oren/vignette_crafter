"""
Pipeline step functions for vignette creation and validation.
Each step receives/returns a state dict.
"""
import logging
from agents.vignette_crafter_agent import VignetteCrafterAgent
from agents.zero_shot_vignette_agent import ZeroShotVignetteAgent
from agents.validator_agent import ValidatorAgent
from agents.persona_crafter_agent import PersonaCrafterAgent
from agents.persona_validator_agent import PersonaValidatorAgent
from data.input.input import (extract_aggregated_edges, get_demographics, get_self_report,
                              sample_self_report, resample_demographics_fields)

logger = logging.getLogger(__name__)


def step_craft_persona(label: str, patient_id, llms: dict, max_retries: int, n_items: int = 3) -> dict:
    """Two-stage persona construction with independent retry loops:
      Stage 1 — Sample demographics, validate for internal consistency. Retry on FAIL.
      Stage 2 — PersonaCrafter selects self-report, validate alignment. Retry selection on FAIL.
    """
    agg_edges    = extract_aggregated_edges(patient_id)
    active_nodes = [node for (p, c), w in agg_edges.items() if w > 0 for node in (p, c)]
    active_nodes = list(dict.fromkeys(active_nodes))  # deduplicate, preserve order

    persona_validator = PersonaValidatorAgent(
        name=f"{label}_PersonaValidator", role="PersonaValidator", llm=llms["persona_validator"]
    )

    # ── Stage 1: random demographics sample, then targeted field fixes ───────
    demographics_attempts = []
    demographics = get_demographics(patient_id)  # random initial sample

    for attempt in range(max_retries + 1):
        result = persona_validator.validate_demographics(demographics)
        demographics_attempts.append({
            "attempt":            attempt + 1,
            "passed":             result["passed"],
            "issues":             result["issues"],
            "problematic_fields": result["problematic_fields"],
        })
        if result["passed"]:
            logger.info("[%s] demographics PASS on attempt %d", label, attempt + 1)
            break
        logger.warning("[%s] demographics FAIL on attempt %d/%d — fixing fields: %s",
                       label, attempt + 1, max_retries + 1, result["problematic_fields"])
        demographics = resample_demographics_fields(demographics, result["problematic_fields"])

    # ── Stage 2: random self-report sample, then targeted fixes ──────────────
    selfreport_attempts = []
    self_report = sample_self_report(active_nodes, n_items)  # random initial sample
    persona_crafter = PersonaCrafterAgent(
        name=f"{label}_PersonaCrafter", role="PersonaCrafter", llm=llms["persona_crafter"],
        active_nodes=active_nodes, n_items=n_items,
    )

    for attempt in range(max_retries + 1):
        result = persona_validator.validate_self_report(demographics, self_report)
        selfreport_attempts.append({
            "attempt":           attempt + 1,
            "passed":            result["passed"],
            "issues":            result["issues"],
            "problematic_items": result["problematic_items"],
        })
        if result["passed"]:
            logger.info("[%s] self-report PASS on attempt %d", label, attempt + 1)
            break
        logger.warning("[%s] self-report FAIL on attempt %d/%d — fixing items: %s",
                       label, attempt + 1, max_retries + 1,
                       list(result["problematic_items"].keys()))
        self_report = persona_crafter.fix_self_report(
            demographics, self_report, result["issues"], result["problematic_items"],
        )

    return {
        "agg_edges":                        agg_edges,
        "demographics":                     demographics,
        "self_report":                      self_report,
        "demographics_validation_attempts": demographics_attempts,
        "selfreport_validation_attempts":   selfreport_attempts,
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
        self_report=self_report,
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
