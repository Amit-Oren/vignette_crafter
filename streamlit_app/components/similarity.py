"""
similarity.py — semantic similarity utilities for the Dataset Analysis page.

Embeds vignettes with OpenAI text-embedding-3-small, computes cosine similarity,
reduces to 2D with t-SNE or PCA, and renders an interactive Plotly scatter plot.
"""

import sys
from pathlib import Path

_STREAMLIT_APP_DIR = Path(__file__).parent.parent
if str(_STREAMLIT_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_STREAMLIT_APP_DIR))

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from utils.loader import get_patients, load_patient

# ── Palette for categorical coloring ──────────────────────────────────────

_PALETTE = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#34495e", "#e91e63", "#00bcd4",
]


# ── Data loading ───────────────────────────────────────────────────────────

def load_experiment_vignettes(experiment_dir: Path) -> list[dict]:
    """Return one record per patient that has a completed vignette."""
    records = []
    for p in get_patients(experiment_dir):
        data = load_patient(p["path"])
        vignette = data.get("vignette", "")
        if not vignette:
            continue
        demo = data.get("demographics", {})
        records.append({
            "patient_id":  data.get("patient_id"),
            "vignette":    vignette,
            "trauma_type": demo.get("trauma_type", "Unknown"),
            "gender":      demo.get("gender", "Unknown"),
            "pcl5":        demo.get("pcl5", 0),
            "age":         demo.get("age", 0),
        })
    return records


# ── Embeddings ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Computing embeddings…")
def get_embeddings(texts: tuple[str, ...], model: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Embed texts locally with sentence-transformers. Cached by text content + model."""
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(model)
    return embedder.encode(list(texts), convert_to_numpy=True)


# ── Similarity ─────────────────────────────────────────────────────────────

def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-9)
    return (normalized @ normalized.T).clip(-1.0, 1.0)


def trauma_type_similarity(sim_matrix: np.ndarray, records: list[dict]) -> list[dict]:
    """For each trauma type, compute mean within-group and mean cross-group similarity."""
    trauma_types = sorted({r["trauma_type"] for r in records})
    rows = []
    for tt in trauma_types:
        idx_in  = [i for i, r in enumerate(records) if r["trauma_type"] == tt]
        idx_out = [i for i, r in enumerate(records) if r["trauma_type"] != tt]
        if len(idx_in) < 2:
            within = float("nan")
        else:
            pairs = [(i, j) for i in idx_in for j in idx_in if i < j]
            within = float(np.mean([sim_matrix[i, j] for i, j in pairs]))
        if idx_in and idx_out:
            cross = float(np.mean([sim_matrix[i, j] for i in idx_in for j in idx_out]))
        else:
            cross = float("nan")
        rows.append({
            "Trauma type": tt,
            "n": len(idx_in),
            "Within-group similarity": within,
            "Cross-group similarity": cross,
            "Δ (within − cross)": within - cross if not (np.isnan(within) or np.isnan(cross)) else float("nan"),
        })
    return rows


def extreme_pairs(sim_matrix: np.ndarray, records: list[dict]) -> dict:
    """Return the most similar and most dissimilar pair of vignettes."""
    n = len(records)
    pairs = [(sim_matrix[i, j], i, j) for i in range(n) for j in range(i + 1, n)]
    most_sim  = max(pairs, key=lambda x: x[0])
    least_sim = min(pairs, key=lambda x: x[0])
    return {
        "most_similar":  {"score": most_sim[0],  "a": records[most_sim[1]],  "b": records[most_sim[2]]},
        "least_similar": {"score": least_sim[0], "a": records[least_sim[1]], "b": records[least_sim[2]]},
    }


# ── Dimensionality reduction ───────────────────────────────────────────────

def reduce_2d(embeddings: np.ndarray, method: str, perplexity: int = 10) -> np.ndarray:
    if method == "t-SNE":
        from sklearn.manifold import TSNE
        perp = min(perplexity, len(embeddings) - 1)
        return TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(embeddings)
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=42).fit_transform(embeddings)


# ── Scatter plot ───────────────────────────────────────────────────────────

def build_scatter(
    coords: np.ndarray,
    records: list[dict],
    color_by: str,
) -> go.Figure:
    hover_texts = [
        f"<b>Patient {r['patient_id']}</b>  ·  {r['trauma_type']}<br>"
        f"{r['gender']}, {r['age']} y/o  ·  PCL-5: {r['pcl5']}<br>"
        f"<i>Click to read full vignette →</i>"
        for r in records
    ]
    # customdata carries patient_id so the page can identify clicked points
    customdata = [[r["patient_id"]] for r in records]

    _continuous = {"pcl5", "age"}
    values = [r[color_by] for r in records]

    if color_by in _continuous:
        fig = go.Figure(go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers",
            customdata=customdata,
            marker=dict(
                size=13,
                color=values,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=color_by.upper()),
                line=dict(width=1, color="#2c3e50"),
            ),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
        ))
    else:
        categories = sorted(set(values))
        fig = go.Figure()
        for i, cat in enumerate(categories):
            mask = np.array([v == cat for v in values])
            fig.add_trace(go.Scatter(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                name=cat,
                customdata=[cd for cd, m in zip(customdata, mask) if m],
                marker=dict(
                    size=13,
                    color=_PALETTE[i % len(_PALETTE)],
                    line=dict(width=1, color="#2c3e50"),
                ),
                text=[t for t, m in zip(hover_texts, mask) if m],
                hovertemplate="%{text}<extra></extra>",
            ))

    fig.update_layout(
        height=580,
        margin=dict(l=40, r=40, t=30, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(title="Dim 1", showgrid=True, gridcolor="#eee", zeroline=False),
        yaxis=dict(title="Dim 2", showgrid=True, gridcolor="#eee", zeroline=False),
        legend=dict(title=dict(text=color_by.replace("_", " ").title()), itemsizing="constant"),
        hovermode="closest",
    )
    return fig
