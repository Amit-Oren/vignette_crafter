"""
6_Dataset_Analysis.py — semantic similarity across all vignettes in an experiment.

Left panel : 2D scatter plot (t-SNE / PCA). Click any point to read the vignette.
Right panel: full vignette reader for the selected patient.
"""

import sys
from pathlib import Path

_STREAMLIT_APP_DIR = Path(__file__).parent.parent
if str(_STREAMLIT_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_STREAMLIT_APP_DIR))

import numpy as np
import streamlit as st

from utils.loader import get_experiments
from components.similarity import (
    load_experiment_vignettes,
    get_embeddings,
    cosine_similarity_matrix,
    trauma_type_similarity,
    extreme_pairs,
    reduce_2d,
    build_scatter,
)

st.set_page_config(page_title="Dataset Analysis", layout="wide")
st.title("Dataset Analysis")
st.markdown("Semantic similarity across all generated vignettes — are they diverse enough?")

# ── Sidebar controls ───────────────────────────────────────────────────────

experiments = get_experiments()
if not experiments:
    st.info("No experiments found. Run a simulation first.")
    st.stop()

exp_names   = [e["name"] for e in experiments]
selected_name = st.sidebar.selectbox("Experiment", exp_names)
experiment_dir = next(e["path"] for e in experiments if e["name"] == selected_name)

st.sidebar.divider()

color_by = st.sidebar.selectbox(
    "Color by",
    options=["trauma_type", "gender", "pcl5", "age"],
    format_func=lambda x: x.replace("_", " ").title(),
)
embed_model = st.sidebar.selectbox(
    "Embedding model",
    options=["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
    help="MiniLM: fast & light. mpnet: slower but higher quality.",
)
method = st.sidebar.selectbox("Reduction method", ["t-SNE", "PCA"])
perplexity = 10
if method == "t-SNE":
    perplexity = st.sidebar.slider(
        "t-SNE perplexity", min_value=5, max_value=30, value=10,
        help="Lower = tighter local clusters. Should be well below n_vignettes.",
    )

# ── Load & embed ───────────────────────────────────────────────────────────

records = load_experiment_vignettes(experiment_dir)
if len(records) < 3:
    st.warning(f"Need at least 3 completed vignettes — found {len(records)}.")
    st.stop()

st.caption(f"{len(records)} vignettes loaded from **{selected_name}**")

texts      = tuple(r["vignette"] for r in records)
embeddings = get_embeddings(texts, embed_model)

# ── Similarity stats ───────────────────────────────────────────────────────

sim_matrix = cosine_similarity_matrix(embeddings)
upper      = sim_matrix[np.triu_indices(len(sim_matrix), k=1)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Mean similarity",   f"{upper.mean():.3f}")
c2.metric("Median similarity", f"{np.median(upper):.3f}")
c3.metric("Min similarity",    f"{upper.min():.3f}", help="Most dissimilar pair")
c4.metric("Max similarity",    f"{upper.max():.3f}", help="Most similar pair")

if upper.mean() > 0.92:
    st.warning("Mean similarity is very high — vignettes may be too similar for a useful training dataset.")
elif upper.mean() > 0.85:
    st.info("Mean similarity is moderate. Some diversity exists but vignettes share a common structure.")
else:
    st.success("Good diversity — vignettes are semantically distinct.")

st.divider()

# ── Two-column layout ──────────────────────────────────────────────────────

col_plot, col_reader = st.columns([3, 2], gap="large")

# resolve selected patient_id from click event or fallback selectbox
pid_options = [str(r["patient_id"]) for r in records]

with col_plot:
    st.subheader(f"{method} projection  ·  colored by {color_by.replace('_', ' ').title()}")

    coords = reduce_2d(embeddings, method, perplexity)
    fig    = build_scatter(coords, records, color_by)

    event  = st.plotly_chart(fig, use_container_width=True,
                              key="scatter_chart", on_select="rerun")

    # Extract patient_id from click if available
    clicked_pid = None
    try:
        points = event["selection"]["points"]
        if points:
            clicked_pid = str(points[0]["customdata"][0])
    except (KeyError, IndexError, TypeError):
        pass

    st.caption("Click any point to read its vignette →")

with col_reader:
    st.subheader("Vignette reader")

    # Sync selectbox with click; clicking updates it, manual change also works
    default_idx = pid_options.index(clicked_pid) if clicked_pid in pid_options else 0
    selected_pid = st.selectbox(
        "Patient", pid_options,
        index=default_idx,
        format_func=lambda p: f"Patient {p}",
        key="vignette_selector",
    )

    record = next(r for r in records if str(r["patient_id"]) == selected_pid)

    # Demographics summary
    st.markdown(
        f"**Trauma:** {record['trauma_type']}  &nbsp;|&nbsp;  "
        f"**Gender:** {record['gender']}  &nbsp;|&nbsp;  "
        f"**Age:** {record['age']}  &nbsp;|&nbsp;  "
        f"**PCL-5:** {record['pcl5']}"
    )
    st.divider()
    st.markdown(record["vignette"])

st.divider()

# ── Insight 1: within vs cross trauma type similarity ─────────────────────

import pandas as pd
from components.similarity import trauma_type_similarity, extreme_pairs

st.subheader("Does trauma type drive the clusters?")
st.markdown(
    "If the model writes differently per trauma type, "
    "**within-group similarity should be higher than cross-group**. "
    "A positive Δ means same-trauma vignettes are more alike than cross-trauma ones."
)

tt_rows = trauma_type_similarity(sim_matrix, records)
df_tt   = pd.DataFrame(tt_rows)

def _color_delta(val):
    if pd.isna(val):
        return ""
    return "color: green" if val > 0.01 else ("color: red" if val < -0.01 else "")

st.dataframe(
    df_tt.style
        .format({"Within-group similarity": "{:.3f}", "Cross-group similarity": "{:.3f}",
                 "Δ (within − cross)": "{:+.3f}"}, na_rep="—")
        .map(_color_delta, subset=["Δ (within − cross)"]),
    use_container_width=True,
    hide_index=True,
)

positive = sum(1 for r in tt_rows if not pd.isna(r["Δ (within − cross)"]) and r["Δ (within − cross)"] > 0.01)
total    = sum(1 for r in tt_rows if not pd.isna(r["Δ (within − cross)"]))
if total > 0:
    if positive / total >= 0.7:
        st.success(f"{positive}/{total} trauma types show higher within-group similarity — "
                   "the model captures trauma-specific language well.")
    elif positive / total >= 0.4:
        st.info(f"{positive}/{total} trauma types show higher within-group similarity — "
                "partial trauma-specific differentiation.")
    else:
        st.warning("Most trauma types show no within-group clustering — "
                   "vignettes may be too structurally similar regardless of trauma type.")

st.divider()

# ── Insight 2: extreme pairs ───────────────────────────────────────────────

st.subheader("Most similar and most different pair")

pairs = extreme_pairs(sim_matrix, records)

tab_sim, tab_diff = st.tabs(["Most similar pair", "Most different pair"])

with tab_sim:
    p = pairs["most_similar"]
    st.markdown(f"**Similarity score: {p['score']:.3f}**")
    c1, c2 = st.columns(2)
    with c1:
        a = p["a"]
        st.markdown(f"**Patient {a['patient_id']}** — {a['trauma_type']}, {a['gender']}, PCL-5 {a['pcl5']}")
        st.markdown(a["vignette"][:600] + "…")
    with c2:
        b = p["b"]
        st.markdown(f"**Patient {b['patient_id']}** — {b['trauma_type']}, {b['gender']}, PCL-5 {b['pcl5']}")
        st.markdown(b["vignette"][:600] + "…")

with tab_diff:
    p = pairs["least_similar"]
    st.markdown(f"**Similarity score: {p['score']:.3f}**")
    c1, c2 = st.columns(2)
    with c1:
        a = p["a"]
        st.markdown(f"**Patient {a['patient_id']}** — {a['trauma_type']}, {a['gender']}, PCL-5 {a['pcl5']}")
        st.markdown(a["vignette"][:600] + "…")
    with c2:
        b = p["b"]
        st.markdown(f"**Patient {b['patient_id']}** — {b['trauma_type']}, {b['gender']}, PCL-5 {b['pcl5']}")
        st.markdown(b["vignette"][:600] + "…")
