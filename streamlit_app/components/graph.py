"""
graph.py — Plotly figure for the PTSD cognitive model graph.

The five PTSD construct nodes are placed in a circular layout and connected
by directed, weighted edges drawn as Plotly annotations (arrows).
"""

import sys
from pathlib import Path

# Ensure streamlit_app/ is importable when this module is used standalone
_STREAMLIT_APP_DIR = Path(__file__).parent.parent
if str(_STREAMLIT_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_STREAMLIT_APP_DIR))

import plotly.graph_objects as go

# ── Constants ──────────────────────────────────────────────────────────────

NODES = [
    "Triggers",
    "Memory",
    "Negative Appraisals",
    "Threat",
    "Maladaptive Strategies",
]

# Fixed positions matching the Ehlers & Clark model layout
NODE_POSITIONS = {
    "Memory":                 (-0.7,  0.6),
    "Negative Appraisals":   ( 0.7,  0.6),
    "Triggers":              ( 0.0,  0.1),
    "Threat":                ( 0.15, -0.35),
    "Maladaptive Strategies":( 0.0, -0.85),
}

# Colours
COLOR_ACTIVE   = "#2ecc71"   # green  — obs_nodes value = 1
COLOR_INACTIVE = "#bdc3c7"   # gray   — obs_nodes value = 0
COLOR_DEFAULT  = "#8ec8ef"   # blue   — obs_nodes not provided

EDGE_COLOR = "#53717f"


def render_cognitive_graph(
    agg_edges: dict,
    obs_nodes: dict = None,  # kept for backwards compatibility, not used for colouring
) -> go.Figure:
    """
    Build and return a Plotly figure of the PTSD cognitive model.

    Parameters
    ----------
    agg_edges : dict
        Edge weights keyed by "Node A -- Node B" (as stored in the patient
        JSON).  Only edges with weight > 0 are drawn.
    obs_nodes : dict, optional
        Node activity flags keyed by node name.  Value 1 = active (green),
        0 = inactive (gray).  When None, all nodes are drawn in blue.

    Returns
    -------
    go.Figure
    """
    node_pos = NODE_POSITIONS

    # ── Node colours ──────────────────────────────────────────────────────
    node_colors = [COLOR_DEFAULT for _ in NODES]

    # ── Node scatter trace ────────────────────────────────────────────────
    node_x = [node_pos[n][0] for n in NODES]
    node_y = [node_pos[n][1] for n in NODES]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(
            size=70,
            color=node_colors,
            line=dict(width=2, color="#2c3e50"),
        ),
        text=[n.replace(" ", "<br>") for n in NODES],
        textposition="middle center",
        textfont=dict(size=12, color="#2c3e50"),
        hovertext=NODES,
        hoverinfo="text",
    )

    # ── Edge annotations (directed arrows) ───────────────────────────────
    import math as _math

    annotations = []
    edge_label_traces_x: list[float] = []
    edge_label_traces_y: list[float] = []
    edge_label_texts: list[str] = []

    # Parse all active edges first so we can detect bidirectional pairs
    active_edges: list[tuple[str, str, float]] = []
    for edge_key, weight in agg_edges.items():
        if weight <= 0:
            continue
        sep = " -- " if " -- " in edge_key else "--"
        parts = edge_key.split(sep, 1)
        if len(parts) != 2:
            continue
        src_name, tgt_name = parts[0].strip(), parts[1].strip()
        if src_name not in node_pos or tgt_name not in node_pos:
            continue
        active_edges.append((src_name, tgt_name, weight))

    active_pairs = {(s, t) for s, t, _ in active_edges}

    for src_name, tgt_name, weight in active_edges:
        x0, y0 = node_pos[src_name]
        x1, y1 = node_pos[tgt_name]

        dx, dy = x1 - x0, y1 - y0
        dist = _math.hypot(dx, dy) or 1e-9

        # Perpendicular unit vector (rotated 90° left)
        perp_x = -(dy / dist)
        perp_y =  (dx / dist)

        # If the reverse edge also exists, offset both arrows sideways
        # so they don't overlap; otherwise draw straight
        lateral = 0.06 if (tgt_name, src_name) in active_pairs else 0.0

        node_r = 0.13   # approx node radius in axis units
        ax = x0 + (dx / dist) * node_r + perp_x * lateral
        ay = y0 + (dy / dist) * node_r + perp_y * lateral
        ex = x1 - (dx / dist) * node_r + perp_x * lateral
        ey = y1 - (dy / dist) * node_r + perp_y * lateral

        arrow_width = max(0.5, min(2.5, weight * 2.5))

        annotations.append(
            dict(
                ax=ax, ay=ay, axref="x", ayref="y",
                x=ex,  y=ey,  xref="x",  yref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=0.8,
                arrowwidth=arrow_width,
                arrowcolor=EDGE_COLOR,
            )
        )

        # Weight label at midpoint, offset perpendicularly
        mx = (ax + ex) / 2 + perp_x * 0.06
        my = (ay + ey) / 2 + perp_y * 0.06
        edge_label_traces_x.append(mx)
        edge_label_traces_y.append(my)
        edge_label_texts.append(f"{weight:.2f}")

    edge_label_trace = go.Scatter(
        x=edge_label_traces_x,
        y=edge_label_traces_y,
        mode="text",
        text=edge_label_texts,
        textfont=dict(size=9, color="#555"),
        hoverinfo="none",
    )

    # ── Figure layout ─────────────────────────────────────────────────────
    fig = go.Figure(data=[edge_label_trace, node_trace])
    fig.update_layout(
        annotations=annotations,
        showlegend=False,
        height=650,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(
            visible=False,
            range=[-1.3, 1.3],
        ),
        yaxis=dict(
            visible=False,
            range=[-1.2, 1.0],
            scaleanchor="x",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
    )

    return fig
