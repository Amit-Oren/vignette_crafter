"""
4_Vignette.py — final vignette, validation attempts, and token usage.
"""

import sys
from pathlib import Path

_STREAMLIT_APP_DIR = Path(__file__).parent.parent
if str(_STREAMLIT_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_STREAMLIT_APP_DIR))

import streamlit as st

from components.sidebar import render_sidebar_selector

st.set_page_config(page_title="Vignette", layout="wide")
st.title("Vignette")

experiment_dir, patient_data = render_sidebar_selector()

if patient_data is None:
    st.info("Select an experiment and patient from the sidebar to begin.")
    st.stop()

vignette          = patient_data.get("vignette", "")
vignette_attempts = patient_data.get("vignette_attempts", [])
val_summary       = patient_data.get("validation_summary", {})
token_usage       = patient_data.get("token_usage", {})
patient_id        = patient_data.get("patient_id", "?")

st.subheader(f"Patient {patient_id}")

# ── Token usage ────────────────────────────────────────────────────────────

if token_usage:
    t1, t2, t3 = st.columns(3)
    t1.metric("Input tokens",  f"{token_usage.get('input',  0):,}")
    t2.metric("Output tokens", f"{token_usage.get('output', 0):,}")
    t3.metric("Total tokens",  f"{token_usage.get('total',  0):,}")

# ── Validation summary ─────────────────────────────────────────────────────

st.divider()
st.subheader("Validation Summary")

passed_flag  = val_summary.get("ultimately_passed", None)
attempts     = val_summary.get("attempts",  "N/A")
passed_count = val_summary.get("passed",    "N/A")
failed_count = val_summary.get("failed",    "N/A")

if passed_flag is True:
    status_label, status_color = "PASSED", "#2ecc71"
elif passed_flag is False:
    status_label, status_color = "FAILED", "#e74c3c"
else:
    status_label, status_color = "N/A", "#bdc3c7"

summary_col, metrics_col = st.columns([1, 3])

with summary_col:
    st.markdown(
        f"""<div style="border:2px solid {status_color};border-radius:8px;
            padding:16px;text-align:center;">
            <span style="font-size:22px;font-weight:bold;color:{status_color};">
                {status_label}
            </span></div>""",
        unsafe_allow_html=True,
    )

with metrics_col:
    m1, m2, m3 = st.columns(3)
    m1.metric("Attempts", attempts)
    m2.metric("Passed",   passed_count)
    m3.metric("Failed",   failed_count)

# ── Final vignette ─────────────────────────────────────────────────────────

st.divider()
st.subheader("Final Vignette")

if vignette:
    st.markdown(
        f"""<div style="background:#f0f4f8;border-left:4px solid #3498db;
            padding:20px 24px;border-radius:6px;font-size:16px;line-height:1.7;">
            {vignette}</div>""",
        unsafe_allow_html=True,
    )
else:
    st.info("No vignette available.")

# ── Attempt details ────────────────────────────────────────────────────────

if vignette_attempts:
    st.divider()
    st.subheader("Validation Attempts")

    for i, att in enumerate(vignette_attempts, start=1):
        passed   = att.get("passed", False)
        feedback = att.get("feedback") or ""
        text     = att.get("vignette", "")
        icon     = "✅" if passed else "❌"
        label    = "PASS" if passed else "FAIL"
        color    = "#2ecc71" if passed else "#e74c3c"

        with st.expander(f"{icon} Attempt {i} — {label}", expanded=(not passed)):
            if text:
                st.markdown(
                    f"""<div style="background:#f9f9f9;border-left:3px solid {color};
                        padding:14px 20px;font-size:15px;line-height:1.7;border-radius:4px;
                        margin-bottom:12px;">{text}</div>""",
                    unsafe_allow_html=True,
                )
            if feedback:
                st.markdown("**Validator feedback:**")
                st.markdown(
                    f"""<div style="background:#fef9e7;border-left:3px solid #f39c12;
                        padding:14px 20px;font-size:14px;line-height:1.6;border-radius:4px;">
                        {feedback}</div>""",
                    unsafe_allow_html=True,
                )
