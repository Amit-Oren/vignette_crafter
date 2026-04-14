import logging
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from agents.base_agent import set_context_subdir, reset_run_tokens, get_run_tokens
from agents.validator_agent import ValidatorAgent
from data.input.input import get_available_ids
from simulation.factory import build_llm
from simulation.output import to_serializable, write_json, write_txt
from simulation.pipelines import PIPELINES
from simulation.steps import step_persona, step_validate_persona, step_zero_shot, step_craft_persona

logger = logging.getLogger(__name__)


class SimulationRunner:
    def __init__(self, num_patients: int, seed: int,
                 models: dict,
                 experiment_dir: Path,
                 patient_ids: list = None,
                 max_retries: int = 2, temperature: float = 0.7,
                 pipeline: str = "vignette", persona_context: bool = False,
                 use_formulation: bool = True, n_items: int = 3):
        self.num_patients    = num_patients
        self.seed            = seed
        self.patient_ids     = patient_ids
        self.max_retries     = max_retries
        self.n_items         = n_items
        self.use_formulation = use_formulation
        self.experiment_dir = experiment_dir
        self.pipeline_name  = pipeline
        self.pipeline       = PIPELINES[pipeline]
        self.persona_context = persona_context
        self.models         = models
        self.temperature    = temperature
        np.random.seed(seed)
        random.seed(seed)
        self._llms = {role: build_llm(model_name, temperature) for role, model_name in models.items()}

    def _sample_patients(self) -> list:
        available = get_available_ids()
        return list(np.random.choice(available, size=self.num_patients, replace=False))

    def run(self):
        patient_ids = self.patient_ids if self.patient_ids is not None else self._sample_patients()

        for patient_id in patient_ids:
            label = f"Client_{patient_id}"
            logger.info("=== Patient %s starting ===", patient_id)
            set_context_subdir(f"patient_{patient_id}")
            reset_run_tokens()

            state: dict = {}
            validator = ValidatorAgent(
                name=f"{label}_Validator", role="Validator", llm=self._llms["validator"],
            )

            for step in self.pipeline:
                logger.info("[%s] step_%s: starting", label, step)
                if step == "craft_persona":
                    state.update(step_craft_persona(label, patient_id, self._llms["validator"], self.max_retries, self.n_items))
                elif step == "persona":
                    state.update(step_persona(label, patient_id, self._llms, self.persona_context, state, self.use_formulation))
                elif step == "validate_vignette":
                    state.update(step_validate_persona(label, state, validator, self.max_retries))
                elif step == "zero_shot":
                    state.update(step_zero_shot(label, patient_id, self._llms))
                logger.info("[%s] step_%s: done", label, step)

            self._save(patient_id, state)

    def _validation_summary(self, attempts: list) -> dict:
        passed = sum(1 for a in attempts if a["passed"])
        failed = sum(1 for a in attempts if not a["passed"])
        ultimately_passed = attempts[-1]["passed"] if attempts else None
        return {"attempts": len(attempts), "passed": passed, "failed": failed,
                "ultimately_passed": ultimately_passed}

    def _save(self, patient_id, state: dict):
        vignette_attempts = state.get("vignette_attempts", [])

        output = to_serializable({
            "patient_id":            patient_id,
            "experiment_timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "pipeline":    self.pipeline_name,
                "models":      self.models,
                "temperature": self.temperature,
                "max_retries": self.max_retries,
                "self_report_items": self.n_items,
                "use_formulation":   self.use_formulation,
                "seed":        self.seed,
            },
            "demographics":               state.get("demographics", {}),
            "self_report":                state.get("self_report", {}),
            "agg_edges":                  state.get("agg_edges", {}),
            "input_validation_attempts":  state.get("input_validation_attempts", []),
            "validation_summary":         self._validation_summary(vignette_attempts),
            "token_usage":                get_run_tokens(),
            "vignette":                   state.get("vignette", ""),
            "vignette_attempts":          vignette_attempts,
        })

        base = self.experiment_dir / f"experiment_{patient_id}"
        with open(f"{base}.json", "w", encoding="utf-8") as f:
            write_json(output.copy(), f)
        write_txt(output, f"{base}.txt")

        logger.info("[Client_%s] saved → %s.json / .txt", patient_id, base.name)
