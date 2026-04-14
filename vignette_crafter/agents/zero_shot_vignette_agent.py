import logging
from .base_agent import BaseAgent
from configs.prompts import ZERO_SHOT_VIGNETTE_PROMPT

logger = logging.getLogger(__name__)


class ZeroShotVignetteAgent(BaseAgent):
    """Generates a vignette in a single prompt with no patient-specific context."""

    def __init__(self, name: str, role: str, llm):
        super().__init__(name, role, system_prompt="", llm=llm)

    def create_vignette(self) -> str:
        vignette = self.respond(ZERO_SHOT_VIGNETTE_PROMPT)
        logger.info("[%s] zero-shot vignette written", self.name)
        return vignette
