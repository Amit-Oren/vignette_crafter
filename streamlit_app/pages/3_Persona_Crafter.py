"""
3_Persona_Crafter.py — demographics, self-report items, and cognitive
formulation for the selected patient.
"""

import sys
from pathlib import Path

_STREAMLIT_APP_DIR = Path(__file__).parent.parent
if str(_STREAMLIT_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_STREAMLIT_APP_DIR))

import pandas as pd
import streamlit as st

from components.graph import render_cognitive_graph
from components.sidebar import render_sidebar_selector

st.set_page_config(page_title="Persona Crafter", layout="wide")
st.title("Persona Crafter")

experiment_dir, patient_data = render_sidebar_selector()

if patient_data is None:
    st.info("Select an experiment and patient from the sidebar to begin.")
    st.stop()

demo        = patient_data.get("demographics", {})
self_report = patient_data.get("self_report", {})
agg_edges   = patient_data.get("agg_edges", {})
patient_id  = patient_data.get("patient_id", "?")
iv_attempts = patient_data.get("input_validation_attempts", [])

st.subheader(f"Patient {patient_id}")

# ── Demographics ───────────────────────────────────────────────────────────

st.markdown("#### Demographics")

demo_fields = [
    ("Age",                 demo.get("age", "N/A")),
    ("Gender",              demo.get("gender", "N/A")),
    ("Nationality",         demo.get("nationality", "N/A")),
    ("Relationship Status", demo.get("relationship_status", "N/A")),
    ("Trauma Type",         demo.get("trauma_type", "N/A")),
    ("PCL-5",               demo.get("pcl5", "N/A")),
]

demo_cols = st.columns(len(demo_fields))
for col, (label, value) in zip(demo_cols, demo_fields):
    col.metric(label, value)

# ── Self-report items ──────────────────────────────────────────────────────

st.divider()
st.markdown("#### Self-Report Items")

NODE_ORDER = [
    "Triggers",
    "Memory",
    "Negative Appraisals",
    "Threat",
    "Maladaptive Strategies",
]

if self_report:
    # Build per-node expanders
    for node in NODE_ORDER:
        items = self_report.get(node, [])
        if not items:
            continue
        with st.expander(f"**{node}**  ({len(items)} item{'s' if len(items) != 1 else ''})", expanded=True):
            rows = []
            for item in items:
                if isinstance(item, dict):
                    rows.append({"Item": item.get("key", ""), "Description": item.get("value", "")})
                else:
                    rows.append({"Item": str(item), "Description": ""})
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
                column_config={"Description": st.column_config.TextColumn(width="large")},
            )
else:
    st.info("No self-report data available for this patient.")

# ── Input validation (Persona Crafter LLM check) ───────────────────────────

if iv_attempts:
    st.divider()
    st.markdown("#### Persona Crafter Validation")
    for att in iv_attempts:
        status = "PASS" if att.get("passed") else "FAIL"
        icon   = "✅" if att.get("passed") else "❌"
        with st.expander(f"{icon} Attempt {att.get('attempt', '?')} — {status}", expanded=False):
            st.markdown(att.get("reasoning", ""))

# ── Cognitive formulation ──────────────────────────────────────────────────

st.divider()
st.subheader("Cognitive Formulation (Aggregated EMA)")

graph_col, table_col = st.columns([2, 1])

with graph_col:
    if agg_edges:
        fig = render_cognitive_graph(agg_edges)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No aggregated edge data available.")

with table_col:
    if agg_edges:
        st.markdown("**Edge weights**")
        edge_rows = [
            {"Edge": k.replace(" -- ", " → "), "Weight": round(v, 4)}
            for k, v in agg_edges.items()
        ]
        active   = [r for r in edge_rows if r["Weight"] > 0]
        inactive = [r for r in edge_rows if r["Weight"] == 0]

        if active:
            st.markdown("*Active*")
            st.dataframe(pd.DataFrame(active),   use_container_width=True, hide_index=True)
        if inactive:
            st.markdown("*Inactive*")
            st.dataframe(pd.DataFrame(inactive), use_container_width=True, hide_index=True)
