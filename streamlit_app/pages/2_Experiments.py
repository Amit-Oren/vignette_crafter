"""
2_Experiments.py — browse all past experiment runs and their patients.
"""

import sys
from pathlib import Path

_STREAMLIT_APP_DIR = Path(__file__).parent.parent
if str(_STREAMLIT_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_STREAMLIT_APP_DIR))

import pandas as pd
import streamlit as st

from utils.loader import get_experiments, get_patients, load_patient

st.set_page_config(page_title="Experiments", layout="wide")
st.title("Experiments")


def _build_patient_table(experiment_dir: Path) -> pd.DataFrame:
    rows = []
    for p in get_patients(experiment_dir):
        try:
            data = load_patient(p["path"])
        except Exception:
            continue

        demo = data.get("demographics", {})
        vs   = data.get("validation_summary", {})
        tu   = data.get("token_usage", {})

        rows.append({
            "Patient ID":       data.get("patient_id", p["patient_id"]),
            "Age":              demo.get("age", ""),
            "Gender":           demo.get("gender", ""),
            "Trauma Type":      demo.get("trauma_type", ""),
            "PCL-5":            demo.get("pcl5", ""),
            "Attempts":         vs.get("attempts", ""),
            "Passed":           "Yes" if vs.get("ultimately_passed") else "No",
            "Tokens":           tu.get("total", ""),
            "_path":            str(p["path"]),
            "_patient_id":      p["patient_id"],
        })
    return pd.DataFrame(rows)


from utils.loader import PROJECT_ROOT
st.write("DEBUG — PROJECT_ROOT:", str(PROJECT_ROOT))
st.write("DEBUG — output_dir exists:", (PROJECT_ROOT / "data" / "output").exists())

experiments = get_experiments()

if not experiments:
    st.info("No experiments found. Run `python main.py` to generate vignettes.")
    st.stop()

selected_pid = st.session_state.get("patient_id")
if selected_pid:
    st.success(
        f"Patient **{selected_pid}** is selected — "
        "navigate to **Persona Crafter** or **Vignette** to view details."
    )

st.markdown(f"Found **{len(experiments)}** experiment(s).")

for exp in experiments:
    with st.expander(f"**{exp['name']}**  —  {exp['timestamp']}", expanded=False):

        patients_meta = get_patients(exp["path"])
        cfg = {}
        models = {}
        if patients_meta:
            try:
                first  = load_patient(patients_meta[0]["path"])
                c      = first.get("config", {})
                cfg    = {
                    "Pipeline":        c.get("pipeline", ""),
                    "Temperature":     c.get("temperature", ""),
                    "Max retries":     c.get("max_retries", ""),
                    "Self-report N":   c.get("self_report_items", ""),
                    "Use formulation": c.get("use_formulation", ""),
                    "Seed":            c.get("seed", ""),
                }
                models = c.get("models", {})
            except Exception:
                pass

        # Config metrics row
        if cfg:
            meta_cols = st.columns(len(cfg))
            for col, (k, v) in zip(meta_cols, cfg.items()):
                col.metric(k, str(v))

        if models:
            with st.expander("Models", expanded=False):
                for role, model in models.items():
                    st.markdown(f"- **{role}**: `{model}`")

        st.markdown("---")

        df = _build_patient_table(exp["path"])
        if df.empty:
            st.warning("No patient data found in this experiment.")
            continue

        display_cols = ["Patient ID", "Age", "Gender", "Trauma Type", "PCL-5",
                        "Attempts", "Passed", "Tokens"]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

        st.markdown("**Select a patient:**")
        btn_cols = st.columns(min(len(df), 8))
        for col, (_, row) in zip(btn_cols, df.iterrows()):
            pid   = row["_patient_id"]
            ppath = row["_path"]
            if col.button(f"Patient {pid}", key=f"sel_{exp['name']}_{pid}"):
                patient_data = load_patient(Path(ppath))
                st.session_state["experiment_path"] = str(exp["path"])
                st.session_state["patient_id"]      = pid
                st.session_state["patient_data"]    = patient_data
                st.success(f"Patient **{pid}** selected.")
