"""
5_Vignette_Analysis.py — structured analysis of the vignette against the
Ehlers & Clark cognitive model.  Coming soon.
"""

import sys
from pathlib import Path

_STREAMLIT_APP_DIR = Path(__file__).parent.parent
if str(_STREAMLIT_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_STREAMLIT_APP_DIR))

import streamlit as st

from components.sidebar import render_sidebar_selector

st.set_page_config(page_title="Vignette Analysis", layout="wide")
st.title("Vignette Analysis")

experiment_dir, patient_data = render_sidebar_selector()

if patient_data is None:
    st.info("Select an experiment and patient from the sidebar to begin.")
    st.stop()

patient_id = patient_data.get("patient_id", "?")
st.subheader(f"Patient {patient_id}")

st.divider()

st.info(
    "Vignette Analysis is coming soon.\n\n"
    "This page will automatically score the generated vignette against the "
    "patient's cognitive formulation — checking which Ehlers & Clark nodes and "
    "edges are present, absent, or incorrectly represented, and comparing them "
    "against the ground-truth EMA edge weights."
)

st.markdown(
    """
    **Planned features:**

    - Per-node detection score (active / inactive)
    - Per-edge detection score with supporting quotes from the vignette
    - Comparison table: detected vs. ground-truth (aggregated EMA)
    - Side-by-side cognitive model graphs (detected vs. ground-truth)
    - Accuracy / Precision / Recall metrics across nodes and edges
    """
)
