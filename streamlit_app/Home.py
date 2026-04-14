"""
Home.py — Vignette Crafter landing page.
"""

import sys
from pathlib import Path

_STREAMLIT_APP_DIR = Path(__file__).parent
if str(_STREAMLIT_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_STREAMLIT_APP_DIR))

import streamlit as st
from utils.loader import get_experiments, get_patients

st.set_page_config(
    page_title="Vignette Crafter",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Header ─────────────────────────────────────────────────────────────────

st.title("Vignette Crafter")
st.markdown(
    "**AI-powered clinical vignette generation for PTSD research, "
    "grounded in Ehlers & Clark's (2000) cognitive model.**"
)

st.divider()

# ── What is it ─────────────────────────────────────────────────────────────

col_what, col_model = st.columns(2, gap="large")

with col_what:
    st.subheader("What it does")
    st.markdown(
        """
        The Vignette Crafter generates realistic, clinically coherent patient vignettes
        for PTSD assessment and training purposes.

        Given a patient's **aggregated EMA (Ecological Momentary Assessment) data**,
        it samples a demographic profile and relevant self-report items, then uses a
        language model to write a first-person vignette that faithfully reflects the
        patient's unique cognitive formulation.

        Each vignette is automatically **validated** against the patient's cognitive
        graph before being accepted.
        """
    )

with col_model:
    st.subheader("Ehlers & Clark Cognitive Model")
    st.markdown(
        """
        The model represents PTSD through five interconnected constructs:

        | Node | Role |
        |------|------|
        **Triggers** | Situational or internal cues that activate trauma memory |
        **Memory** | Fragmented, poorly contextualised trauma memories |
        **Negative Appraisals** | Distorted meanings assigned to the trauma or its aftermath |
        **Threat** | Persistent sense of current danger |
        **Maladaptive Strategies** | Avoidance, suppression, and safety behaviours |

        Directed edges between nodes capture the **causal relationships** — derived from
        EMA data — that shape each patient's individual formulation.
        """
    )

st.divider()

# ── Pipeline ───────────────────────────────────────────────────────────────

st.subheader("Generation Pipeline")

p1, p2, p3, p4 = st.columns(4, gap="small")

with p1:
    st.markdown(
        """
        **1 — Persona Crafter**

        Selects the most clinically coherent self-report items for each active
        node, guided by the patient's EMA edge strengths and demographics.
        """
    )
with p2:
    st.markdown(
        """
        **2 — Vignette Crafter**

        Writes a first-person clinical vignette that reflects the selected
        demographics, self-report items, and cognitive formulation.
        """
    )
with p3:
    st.markdown(
        """
        **3 — Validator**

        Checks that the vignette correctly reflects the patient's required
        edges (active) and forbidden edges (inactive). Provides feedback for
        revision if it fails.
        """
    )
with p4:
    st.markdown(
        """
        **4 — Retry loop**

        On a FAIL verdict the vignette is revised using the validator's
        feedback. The loop repeats up to `max_retries` times.
        """
    )

st.divider()

# ── Stats ──────────────────────────────────────────────────────────────────

experiments   = get_experiments()
total_patients = sum(len(get_patients(e["path"])) for e in experiments)

c1, c2 = st.columns(2)
c1.metric("Experiments run", len(experiments))
c2.metric("Vignettes generated", total_patients)

st.divider()

# ── Navigation ─────────────────────────────────────────────────────────────

st.subheader("Navigation")
st.markdown(
    """
    | Page | Description |
    |------|-------------|
    | **Experiments** | Browse all past runs, their config, and per-patient results. |
    | **Persona Crafter** | View the sampled demographics, selected self-report items, and cognitive formulation graph for a patient. |
    | **Vignette** | Read the final generated vignette and inspect each validation attempt. |
    | **Vignette Analysis** | *(Coming soon)* Structured analysis of the vignette against the cognitive model. |
    """
)

if experiments:
    st.divider()
    st.subheader("Recent experiments")
    for exp in experiments[:5]:
        n = len(get_patients(exp["path"]))
        st.markdown(f"- **{exp['name']}** — {n} vignette(s)")
else:
    st.info("No experiments found yet. Run `python main.py` to generate your first vignette.")
