import logging
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from agents.base_agent import set_context_subdir, reset_run_tokens, get_run_tokens
from agents.vignette_validator_agent import VignetteValidatorAgent
from simulation.factory import build_llm
from simulation.output import to_serializable, write_json, write_txt
from simulation.pipelines import PIPELINES
from simulation.steps import step_persona, step_validate_persona, step_zero_shot, step_craft_persona, step_load_persona

logger = logging.getLogger(__name__)


class SimulationRunner:
    def __init__(self, num_personas: int, seed: int,
                 models: dict,
                 experiment_dir: Path,
                 persona_ids: list = None,
                 max_retries: int = 2, temperature: float = 0.7,
                 pipeline: str = "vignette", persona_context: bool = False,
                 use_formulation: bool = True, n_items: int = 3, node_prob: float = 0.7,
                 edge_prob: float = 0.5, persona_source: str = None):
        self.num_personas    = num_personas
        self.seed            = seed
        self.persona_ids     = persona_ids
        self.max_retries     = max_retries
        self.n_items         = n_items
        self.node_prob       = node_prob
        self.edge_prob       = edge_prob
        self.use_formulation = use_formulation
        self.persona_source  = persona_source
        self.experiment_dir  = experiment_dir
        self.pipeline_name   = pipeline
        self.pipeline        = PIPELINES[pipeline]
        self.persona_context = persona_context
        self.models          = models
        self.temperature     = temperature
        np.random.seed(seed)
        random.seed(seed)
        self._llms = {role: build_llm(model_name, temperature) for role, model_name in models.items()}

    def _sample_personas(self) -> list:
        return list(range(1, self.num_personas + 1))

    def run(self):
        persona_ids = self.persona_ids if self.persona_ids else self._sample_personas()

        failed = []
        for persona_id in persona_ids:
            label = f"Persona_{persona_id}"
            logger.info("=== Persona %s starting ===", persona_id)
            set_context_subdir(f"persona_{persona_id}")
            reset_run_tokens()

            try:
                state: dict = {}
                validator = VignetteValidatorAgent(
                    name=f"{label}_VignetteValidator", role="VignetteValidator", llm=self._llms["vignette_validator"],
                )

                for step in self.pipeline:
                    logger.info("[%s] step_%s: starting", label, step)
                    if step == "load_persona":
                        if not self.persona_source:
                            raise ValueError("pipeline 'vignette_from_persona' requires persona_source in config")
                        state.update(step_load_persona(label, persona_id, self.persona_source))
                    elif step == "craft_persona":
                        state.update(step_craft_persona(label, persona_id, self._llms, self.max_retries, self.n_items, self.node_prob, self.edge_prob))
                    elif step == "persona":
                        state.update(step_persona(label, persona_id, self._llms, self.persona_context, state, self.use_formulation, self.node_prob, self.edge_prob))
                    elif step == "validate_vignette":
                        state.update(step_validate_persona(label, state, validator, self.max_retries))
                    elif step == "zero_shot":
                        state.update(step_zero_shot(label, persona_id, self._llms, self.node_prob, self.edge_prob))
                    logger.info("[%s] step_%s: done", label, step)

                self._save(persona_id, state)
            except Exception as e:
                logger.error("=== Persona %s FAILED: %s — skipping ===", persona_id, e, exc_info=True)
                failed.append(persona_id)

        if failed:
            logger.warning("=== Run complete. %d persona(s) failed: %s ===", len(failed), failed)
        else:
            logger.info("=== Run complete. All %d personas succeeded ===", len(persona_ids))

    def _validation_summary(self, attempts: list) -> dict:
        passed = sum(1 for a in attempts if a["passed"])
        failed = sum(1 for a in attempts if not a["passed"])
        ultimately_passed = attempts[-1]["passed"] if attempts else None
        return {"attempts": len(attempts), "passed": passed, "failed": failed,
                "ultimately_passed": ultimately_passed}

    def _save(self, persona_id, state: dict):
        vignette_attempts = state.get("vignette_attempts", [])

        output = to_serializable({
            "persona_id":            persona_id,
            "experiment_timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "pipeline":    self.pipeline_name,
                "models":      self.models,
                "temperature": self.temperature,
                "max_retries": self.max_retries,
                "self_report_items": self.n_items,
                "use_formulation":   self.use_formulation,
                "edge_prob":         self.edge_prob,
                "seed":        self.seed,
            },
            "demographics":               state.get("demographics", {}),
            "self_report":                state.get("self_report", {}),
            "agg_edges":                  state.get("agg_edges", {}),
            "demographics_validation_attempts": state.get("demographics_validation_attempts", []),
            "selfreport_validation_attempts":   state.get("selfreport_validation_attempts", []),
            "validation_summary":         self._validation_summary(vignette_attempts),
            "token_usage":                get_run_tokens(),
            "vignette":                   state.get("vignette", ""),
            "vignette_attempts":          vignette_attempts,
        })

        base = self.experiment_dir / f"experiment_{persona_id}"
        with open(f"{base}.json", "w", encoding="utf-8") as f:
            write_json(output.copy(), f)
        write_txt(output, f"{base}.txt")

        logger.info("[Persona_%s] saved → %s.json / .txt", persona_id, base.name)
