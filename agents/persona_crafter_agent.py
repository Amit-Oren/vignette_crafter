import logging
from pydantic import BaseModel
from .base_agent import BaseAgent
from configs.prompts import PERSONA_CRAFTER_SYSTEM_PROMPT, PERSONA_CRAFTER_USER_PROMPT
from data.input.input import _NODE_POOLS


class NodeSelection(BaseModel):
    component: str      # e.g. "Triggers"
    items: list[str]    # list of replacement key names


class SelectionResult(BaseModel):
    selections: list[NodeSelection]

logger = logging.getLogger(__name__)


class PersonaCrafterAgent(BaseAgent):
    """Fixes flagged self-report items with clinically coherent replacements."""

    def __init__(self, name: str, role: str, llm,
                 active_nodes: list[str], n_items: int = 3):
        self.active_nodes = active_nodes
        self.n_items      = n_items
        super().__init__(name, role, system_prompt="", llm=llm)

    def setup_agent(self):
        return None

    def fix_self_report(self, demographics: dict, self_report: dict,
                        issues: list[dict], problematic_items: dict[str, list[str]]) -> dict:
        """Replace only the flagged items with better alternatives."""
        self.system_prompt = PERSONA_CRAFTER_SYSTEM_PROMPT.format(
            replacement_pools=self.fmt_pools(list(problematic_items.keys())),
        )
        user_prompt = PERSONA_CRAFTER_USER_PROMPT.format(
            demographics=self.fmt_demographics(demographics),
            current_self_report=self.fmt_self_report(self_report),
            issues=self._fmt_issues(issues),
        )

        result: SelectionResult = self._invoke_structured(SelectionResult, self.system_prompt, user_prompt)
        if result is None:
            logger.warning("[%s] fix_self_report parsing failed — keeping original items", self.name)
            return self_report

        selections_by_node = self._normalize_selections(result)
        self.log_response(user_prompt, str(selections_by_node), output=selections_by_node)
        return self._apply_selections(self_report, selections_by_node, problematic_items)

    def _normalize_selections(self, result: SelectionResult) -> dict[str, list[str]]:
        """Parse LLM output, stripping any 'key: value' format down to just the key."""
        return {
            ns.component: [k.split(":")[0].strip() for k in ns.items]
            for ns in result.selections
        }

    def _apply_selections(self, self_report: dict, selections_by_node: dict,
                          problematic_items: dict[str, list[str]]) -> dict:
        """Apply LLM-chosen replacements to self_report, node by node."""
        updated = {node: list(items) for node, items in self_report.items()}
        for node, bad_keys in problematic_items.items():
            pool = _NODE_POOLS.get(node, {})
            if not pool:
                logger.warning("[%s] unknown node '%s' — skipping", self.name, node)
                continue
            current_keys = [i["key"] if isinstance(i, dict) else i for i in updated.get(node, [])]
            merged = self._merge_node(bad_keys, pool, selections_by_node.get(node, []), current_keys)
            updated[node] = [{"key": k, "value": pool[k]} for k in merged]
        logger.info("[%s] fixed items for nodes: %s", self.name, list(problematic_items.keys()))
        return updated

    def _merge_node(self, bad_keys: list, pool: dict, replacements: list, current_keys: list) -> list:
        """Return kept + new replacement keys for a node, padding to n_items if needed."""
        kept     = [k for k in current_keys if k not in bad_keys]
        new_keys = [k for k in replacements if k in pool and k not in kept and k not in bad_keys]
        merged   = kept + new_keys
        for key in pool:
            if len(merged) == self.n_items:
                break
            if key not in merged and key not in bad_keys:
                merged.append(key)
        return merged[:self.n_items]

    @staticmethod
    def fmt_pools(nodes: list) -> str:
        lines = []
        for node in nodes:
            pool = _NODE_POOLS.get(node, {})
            lines.append(f"  {node}:")
            for key, value in pool.items():
                lines.append(f"    - {key}: {value}")
        return "\n".join(lines)
