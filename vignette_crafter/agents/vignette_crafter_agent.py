import logging
from .base_agent import BaseAgent
from data.input.input import get_aggregated_context, sample_demographics
from configs.prompts import (PERSONA_CRAFTER_AGENT_PROMPT,
                             PERSONA_CRAFTER_AGENT_PROMPT_CONTEXT,
                             PERSONA_CRAFTER_RETRY_PROMPT,
                             PERSONA_CRAFTER_USER_PROMPT,
                             NO_FORMULATION_SYSTEM_PROMPT,
                             NO_FORMULATION_USER_PROMPT)

logger = logging.getLogger(__name__)


class VignetteCrafterAgent(BaseAgent):
    """Agent responsible for crafting a persona vignette based on aggregated EMA and randomly sampled demographics."""
    def __init__(self, name: str, role: str, llm, patient_id: int,
                 use_demographics: bool = True, use_self_report: bool = True,
                 use_formulation: bool = True, demographics: dict = None):
        context = get_aggregated_context(patient_id)
        demo = demographics or sample_demographics()

        if not use_formulation:
            # Demographics only — no edges or nodes
            prompt = NO_FORMULATION_SYSTEM_PROMPT.format(
                demographics=self._format_demographics(demo),
            )
            self._user_prompt = NO_FORMULATION_USER_PROMPT
        elif use_demographics or use_self_report:
            required_edges, forbidden_edges = self._split_edges(context["edges"])
            prompt = PERSONA_CRAFTER_AGENT_PROMPT_CONTEXT.format(
                required_edges=required_edges,
                forbidden_edges=forbidden_edges,
                demographics=self._format_demographics(demo) if use_demographics else "  (not provided)",
                self_report=self._format_self_report(context["nodes"]) if use_self_report else "  (not provided)",
            )
            self._user_prompt = PERSONA_CRAFTER_USER_PROMPT
        else:
            required_edges, forbidden_edges = self._split_edges(context["edges"])
            prompt = PERSONA_CRAFTER_AGENT_PROMPT.format(
                required_edges=required_edges,
                forbidden_edges=forbidden_edges,
            )
            self._user_prompt = PERSONA_CRAFTER_USER_PROMPT

        super().__init__(name, role, prompt, llm)

    def create_vignette(self) -> str:
        self.vignette = self.respond(self._user_prompt)
        logger.info("[%s] vignette written", self.name)
        return self.vignette

    def create_vignette_with_feedback(self, feedback: str) -> str:
        retry_prompt = PERSONA_CRAFTER_RETRY_PROMPT.format(
            feedback=feedback,
            previous_persona=self.vignette,
        )
        self.vignette = self.respond(retry_prompt)
        logger.info("[%s] vignette revised after feedback", self.name)
        return self.vignette

    def _split_edges(self, edges: dict) -> tuple[str, str]:
        required = [k for k, v in edges.items() if v.get("strength", 0) > 0]
        forbidden = [k for k, v in edges.items() if v.get("strength", 0) == 0]
        req_str = "\n".join(f"  - {k[0]} → {k[1]}" for k in required) or "  (none)"
        forb_str = "\n".join(f"  - {k[0]} → {k[1]}" for k in forbidden) or "  (none)"
        return req_str, forb_str

    def _format_demographics(self, demo: dict) -> str:
        return "\n".join(f"  {k}: {v}" for k, v in demo.items())

    def _format_self_report(self, nodes: dict) -> str:
        lines = []
        for node, data in nodes.items():
            items = data.get("items", [])
            if items:
                formatted = ", ".join(
                    f"{i['key']}: {i['value']}" if isinstance(i, dict) else i
                    for i in items
                )
                lines.append(f"  {node}: {formatted}")
        return "\n".join(lines)
