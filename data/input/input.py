import random
import pandas as pd
from pathlib import Path
from configs.formulation_config import DIRECT_EDGES
from configs.demographics import AGE, GENDER, NATIONALITY, RELATIONSHIP_STATUS, TRAUMA_TYPE, PCL5
from configs.self_report import (TRIGGERS, NEGATIVE_APPRAISALS, MEMORY,
                                  THREAT, MALADAPTIVE_STRATEGIES,
                                  DISTRACTION_STRATEGIES, AVOIDANCE_STRATEGIES)

DATA_DIR = Path(__file__).parent

_NAME_MAP = {
    "Appraisals": "Negative Appraisals",
    "Strategies": "Maladaptive Strategies",
    "Memory":     "Memory",
    "Threat":     "Threat",
    "Triggers":   "Triggers",
}

_df = pd.read_csv(DATA_DIR / "aggregated_ema_data.csv")

_STRATEGIES_POOL = {**MALADAPTIVE_STRATEGIES, **DISTRACTION_STRATEGIES, **AVOIDANCE_STRATEGIES}

_NODE_POOLS: dict[str, dict] = {
    "Triggers":               TRIGGERS,
    "Negative Appraisals":    NEGATIVE_APPRAISALS,
    "Memory":                 MEMORY,
    "Threat":                 THREAT,
    "Maladaptive Strategies": _STRATEGIES_POOL,
}


# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_demographics() -> dict:
    """Sample a random patient demographics profile."""
    return {
        "age":                 random.randint(AGE["min"], AGE["max"]),
        "gender":              random.choice(GENDER),
        "nationality":         random.choice(NATIONALITY),
        "relationship_status": random.choice(RELATIONSHIP_STATUS),
        "trauma_type":         random.choice(TRAUMA_TYPE),
        "pcl5":                random.randint(PCL5["min"], PCL5["max"]),
    }


def resample_demographics_fields(demographics: dict, fields: list[str]) -> dict:
    """Re-sample only the specified fields, keeping all others unchanged."""
    _samplers = {
        "age":                 lambda: random.randint(AGE["min"], AGE["max"]),
        "gender":              lambda: random.choice(GENDER),
        "nationality":         lambda: random.choice(NATIONALITY),
        "relationship_status": lambda: random.choice(RELATIONSHIP_STATUS),
        "trauma_type":         lambda: random.choice(TRAUMA_TYPE),
        "pcl5":                lambda: random.randint(PCL5["min"], PCL5["max"]),
    }
    updated = dict(demographics)
    for field in fields:
        if field in _samplers:
            updated[field] = _samplers[field]()
    return updated


def sample_self_report(nodes: list[str], n_items: int = 3) -> dict:
    """Sample self-report items for each active node.
    Returns dict of node_name → list of {key, value} dicts.
    """
    result = {}
    for node in nodes:
        pool = _NODE_POOLS.get(node, {})
        sampled = random.sample(list(pool.items()), min(n_items, len(pool)))
        result[node] = [{"key": k, "value": v} for k, v in sampled]
    return result


# ── Per-patient context ───────────────────────────────────────────────────────

def get_demographics(_patient_id) -> dict:
    """Sample a demographics profile for this patient (seeded externally)."""
    return sample_demographics()


def get_self_report(patient_id) -> dict:
    """Return sampled self-report items for each active node for this patient."""
    agg = extract_aggregated_edges(patient_id)
    active_nodes = list(dict.fromkeys(node for (p, c), w in agg.items() if w > 0 for node in (p, c)))
    return {node: items for node, items in sample_self_report(active_nodes).items()}


# ── Edge extraction ───────────────────────────────────────────────────────────

def get_available_ids() -> list:
    """Return list of all patient IDs in the CSV."""
    return _df["id"].unique().tolist()


def extract_aggregated_edges(patient_id) -> dict:
    """Extract aggregated edge strengths for the given patient ID from CSV."""
    rows = _df[_df["id"] == patient_id]
    if rows.empty:
        raise ValueError(f"ID {patient_id!r} not found. Available IDs: {get_available_ids()}")

    edges = DIRECT_EDGES.copy()
    for _, row in rows.iterrows():
        parent = _NAME_MAP[row["Parent"]]
        child  = _NAME_MAP[row["Child"]]
        key = (parent, child)
        if key in edges:
            edges[key] = row["estimate"]

    return edges


# ── Context assembly ──────────────────────────────────────────────────────────

def get_aggregated_context(patient_id) -> dict:
    """Build full context: sampled self-report items + aggregated edge strengths."""
    agg = extract_aggregated_edges(patient_id)

    active_nodes = list(dict.fromkeys(
        node for (parent, child), weight in agg.items() if weight > 0 for node in (parent, child)
    ))

    ni = sample_self_report(active_nodes)

    return {
        "nodes": {node: {"items": items} for node, items in ni.items()},
        "edges": {
            edge: {
                "strength":     round(weight, 4),
                "parent_items": ni.get(edge[0], []),
                "child_items":  ni.get(edge[1], []),
            }
            for edge, weight in agg.items()
        },
    }
