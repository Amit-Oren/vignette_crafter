import logging
from .base_agent import BaseAgent
from data.input.input import sample_formulation, sample_demographics
from configs.prompts import (VIGNETTE_CRAFTER_PROMPT,
                             VIGNETTE_CRAFTER_PROMPT_CONTEXT,
                             VIGNETTE_CRAFTER_RETRY_PROMPT,
                             VIGNETTE_CRAFTER_USER_PROMPT,
                             NO_FORMULATION_SYSTEM_PROMPT,
                             NO_FORMULATION_USER_PROMPT,
                             NO_FORMULATION_SR_SYSTEM_PROMPT,
                             NO_FORMULATION_SR_USER_PROMPT,
                             ZERO_SHOT_VIGNETTE_PROMPT)

logger = logging.getLogger(__name__)


class VignetteCrafterAgent(BaseAgent):
    """Writes a clinical vignette grounded in the patient's EMA formulation."""

    def __init__(self, name: str, role: str, llm,
                 use_demographics: bool = True, use_self_report: bool = True,
                 use_formulation: bool = True, demographics: dict = None,
                 self_report: dict = None, formulation: dict = None,
                 node_prob: float = 0.7, edge_prob: float = 0.5):
        context = formulation or sample_formulation(node_prob=node_prob, edge_prob=edge_prob)
        demo = demographics or sample_demographics()

        if not use_formulation and (use_demographics or use_self_report):
            sr = self_report if self_report is not None else self._nodes_to_self_report(context["nodes"])
            prompt = NO_FORMULATION_SR_SYSTEM_PROMPT.format(
                demographics=self.fmt_demographics(demo) if use_demographics else "  (not provided)",
                self_report=self.fmt_self_report(sr)     if use_self_report  else "  (not provided)",
            )
            self._user_prompt = NO_FORMULATION_SR_USER_PROMPT
        elif not use_formulation:
            # no_context: no demographics, no self-report, no edges
            prompt = ZERO_SHOT_VIGNETTE_PROMPT
            self._user_prompt = "Write a clinical vignette for a patient with PTSD."
        elif use_demographics or use_self_report:
            required_edges, forbidden_edges = self._split_edges(context["edges"])
            sr = self_report if self_report is not None else self._nodes_to_self_report(context["nodes"])
            prompt = VIGNETTE_CRAFTER_PROMPT_CONTEXT
            self._user_prompt = VIGNETTE_CRAFTER_USER_PROMPT.format(
                demographics=self.fmt_demographics(demo) if use_demographics else "  (not provided)",
                self_report=self.fmt_self_report(sr) if use_self_report else "  (not provided)",
                required_edges=required_edges,
                forbidden_edges=forbidden_edges,
            )
        else:
            required_edges, forbidden_edges = self._split_edges(context["edges"])
            prompt = VIGNETTE_CRAFTER_PROMPT.format(
                required_edges=required_edges,
                forbidden_edges=forbidden_edges,
            )
            self._user_prompt = VIGNETTE_CRAFTER_USER_PROMPT.format(
                demographics="  (not provided)",
                self_report="  (not provided)",
                required_edges=required_edges,
                forbidden_edges=forbidden_edges,
            )

        super().__init__(name, role, prompt, llm)

    def create_vignette(self) -> str:
        self.vignette = self.respond(self._user_prompt)
        logger.info("[%s] vignette written", self.name)
        return self.vignette

    def create_vignette_with_feedback(self, feedback: str) -> str:
        retry_prompt = VIGNETTE_CRAFTER_RETRY_PROMPT.format(
            feedback=feedback,
            previous_persona=self.vignette,
        )
        self.vignette = self.respond(retry_prompt)
        logger.info("[%s] vignette revised after feedback", self.name)
        return self.vignette

    def _split_edges(self, edges: dict) -> tuple[str, str]:
        required = [k for k, v in edges.items() if v.get("strength", 0) > 0]
        forbidden = [k for k, v in edges.items() if v.get("strength", 0) == 0]
        req_str  = "\n".join(f"  - {k[0]} → {k[1]}" for k in required)  or "  (none)"
        forb_str = "\n".join(f"  - {k[0]} → {k[1]}" for k in forbidden) or "  (none)"
        return req_str, forb_str

    @staticmethod
    def _nodes_to_self_report(nodes: dict) -> dict:
        """Convert get_aggregated_context nodes format to self_report format."""
        return {node: data.get("items", []) for node, data in nodes.items() if data.get("items")}
