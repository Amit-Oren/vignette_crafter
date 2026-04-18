import logging
from pydantic import BaseModel
from .base_agent import BaseAgent, _count
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
        structured_llm = self.llm.with_structured_output(SelectionResult, include_raw=True, method="function_calling")
        raw = structured_llm.invoke([
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": user_prompt},
        ])
        if isinstance(raw, dict):
            _count(raw.get("raw"))
            result: SelectionResult = raw["parsed"]
        else:
            result: SelectionResult = raw

        if result is None:
            logger.warning("[%s] fix_self_report parsing failed — keeping original items", self.name)
            return self_report

        selections_dict = {ns.component: ns.items for ns in result.selections}
        self.log_response(user_prompt, str(selections_dict), output=selections_dict)

        # Build lookup from component name → list of suggested keys
        # Normalize: strip value part if LLM returns "key: value" format
        selections_by_node: dict[str, list[str]] = {}
        for ns in result.selections:
            selections_by_node[ns.component] = [k.split(":")[0].strip() for k in ns.items]

        # Merge: replace only problematic items, keep the rest
        updated = {node: list(items) for node, items in self_report.items()}
        for node, bad_keys in problematic_items.items():
            pool = _NODE_POOLS.get(node, {})
            if not pool:
                logger.warning("[%s] unknown node '%s' in problematic_items — skipping", self.name, node)
                continue
            replacements = selections_by_node.get(node, [])
            current_keys = [i["key"] if isinstance(i, dict) else i for i in updated.get(node, [])]

            kept     = [k for k in current_keys if k not in bad_keys]
            # Exclude bad_keys from replacements even if the LLM re-suggested them
            new_keys = [k for k in replacements if k in pool and k not in kept and k not in bad_keys]
            merged   = kept + new_keys

            if len(merged) < self.n_items:
                for key in pool:
                    if key not in merged and key not in bad_keys:
                        merged.append(key)
                    if len(merged) == self.n_items:
                        break

            updated[node] = [{"key": k, "value": pool[k]} for k in merged[:self.n_items]]

        logger.info("[%s] fixed items for nodes: %s", self.name, list(problematic_items.keys()))
        return updated

    @staticmethod
    def fmt_pools(nodes: list) -> str:
        lines = []
        for node in nodes:
            pool = _NODE_POOLS.get(node, {})
            lines.append(f"  {node}:")
            for key, value in pool.items():
                lines.append(f"    - {key}: {value}")
        return "\n".join(lines)
