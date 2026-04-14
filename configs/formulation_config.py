"""Ehlers & Clark PTSD formulation — directed causal graph."""

DIRECT_EDGES: dict = {
    ("Triggers", "Maladaptive Strategies"): 0,
    ("Triggers", "Threat"): 0,
    ("Triggers", "Memory"): 0,
    ("Triggers", "Negative Appraisals"): 0,
    ("Negative Appraisals", "Maladaptive Strategies"): 0,
    ("Negative Appraisals", "Threat"): 0,
    ("Negative Appraisals", "Memory"): 0,
    ("Negative Appraisals", "Triggers"): 0,
    ("Memory", "Maladaptive Strategies"): 0,
    ("Memory", "Threat"): 0,
    ("Memory", "Negative Appraisals"): 0,
    ("Memory", "Triggers"): 0,
    ("Threat", "Maladaptive Strategies"): 0,
    ("Threat", "Memory"): 0,
    ("Threat", "Negative Appraisals"): 0,
    ("Threat", "Triggers"): 0,
    ("Maladaptive Strategies", "Memory"): 0,
    ("Maladaptive Strategies", "Negative Appraisals"): 0,
}
