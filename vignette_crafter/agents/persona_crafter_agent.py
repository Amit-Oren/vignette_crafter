import json
import logging
import re
from .base_agent import BaseAgent, _count
from configs.prompts import SELF_REPORT_SELECTOR_PROMPT
from data.input.input import _NODE_POOLS

logger = logging.getLogger(__name__)


class PersonaCrafterAgent(BaseAgent):
    """Selects coherent self-report items for each active node.

    Given the full candidate pools, patient demographics, and aggregated EMA
    edge strengths, the agent picks N items per node that best fit the patient.
    A structural check then ensures valid keys, no duplicates, and correct count.
    """

    def __init__(self, name: str, role: str, llm,
                 active_nodes: list[str], demographics: dict, agg_edges: dict,
                 n_items: int = 3):
        self.active_nodes = active_nodes
        self.demographics = demographics
        self.agg_edges    = agg_edges
        self.n_items      = n_items
        super().__init__(name, role, system_prompt="", llm=llm)

    def select(self) -> dict:
        prompt = SELF_REPORT_SELECTOR_PROMPT.format(
            n_items=self.n_items,
            demographics=self.fmt_demographics(),
            edges=self.fmt_edges(),
            pools=self.fmt_pools(),
        )

        response = self.llm.invoke([{"role": "user", "content": prompt}])
        _count(response)
        raw_text = response.content
        self.log_response(prompt, raw_text)

        # Extract JSON object from response
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        selections = json.loads(match.group()) if match else {}

        logger.info("[%s] selected self-report for: %s", self.name, list(selections.keys()))
        return self.validate(selections)

    # ── Formatting helpers ────────────────────────────────────────────────────

    def fmt_demographics(self) -> str:
        return "\n".join(f"  {k}: {v}" for k, v in self.demographics.items())

    def fmt_edges(self) -> str:
        lines = [
            f"  {p} → {c}: {v:.2f}"
            for (p, c), v in self.agg_edges.items() if v > 0
        ]
        return "\n".join(lines) or "  (none)"

    def fmt_pools(self) -> str:
        lines = []
        for node in self.active_nodes:
            pool = _NODE_POOLS.get(node, {})
            lines.append(f"  {node}:")
            for key, value in pool.items():
                lines.append(f"    - {key}: {value}")
        return "\n".join(lines)

    # ── Structural validation ─────────────────────────────────────────────────

    def validate(self, selections: dict[str, list[str]]) -> dict:
        """Ensure valid keys, no duplicates, and correct item count per node."""
        result = {}
        for node in self.active_nodes:
            pool = _NODE_POOLS.get(node, {})
            raw  = selections.get(node, [])

            # Normalise: accept plain strings or {"key": ..., "value": ...} dicts
            keys = [k["key"] if isinstance(k, dict) else k for k in raw]
            valid = list(dict.fromkeys(k for k in keys if k in pool))

            if len(valid) < self.n_items:
                for key in pool:
                    if key not in valid:
                        valid.append(key)
                    if len(valid) == self.n_items:
                        break

            result[node] = [{"key": k, "value": pool[k]} for k in valid[:self.n_items]]
        return result
