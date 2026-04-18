import logging
from .base_agent import BaseAgent
from configs.prompts import ZERO_SHOT_VIGNETTE_PROMPT

logger = logging.getLogger(__name__)


class ZeroShotVignetteAgent(BaseAgent):
    """Generates a vignette in a single prompt with no patient-specific context."""

    def __init__(self, name: str, role: str, llm):
        super().__init__(name, role, system_prompt=ZERO_SHOT_VIGNETTE_PROMPT, llm=llm)

    def setup_agent(self):
        return None

    def create_vignette(self) -> str:
        vignette = self.respond("Write the vignette.")
        logger.info("[%s] zero-shot vignette written", self.name)
        return vignette
