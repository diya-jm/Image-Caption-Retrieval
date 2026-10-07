"""Interactive results dashboard for the Neural Caption-Image Retrieval project.

Run from the repository root:
    python -m streamlit run app.py

The dashboard focuses on the actual project question:
given a caption, how well do different sentence encoders align it with
image features in a shared embedding space?

It visualizes the reported paper results, our replication results, the
Transformer extension, and available training logs from results/.
"""

from pathlib import Path
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------
# Page / theme
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Caption–Image Retrieval",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        border: 1px solid #374151;
        margin-bottom: 1.2rem;
    }
    .hero h1 {margin: 0 0 .35rem 0; font-size: 2.2rem;}
    .hero p {margin: 0; color: #cbd5e1; font-size: 1rem;}
    .eyebrow {
        color: #60a5fa;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .08em;
        font-size: .78rem;
        margin-bottom: .45rem;
    }
    .section-note {
        color: #64748b;
        font-size: .92rem;
        margin-top: -.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        padding: 1rem 1.05rem;
        border: 1px solid #334155;
        border-radius: 14px;
        background: rgba(30, 41, 59, .45);
        min-height: 105px;
    }
    .metric-label {font-size: .78rem; color: #94a3b8;}
    .metric-value {font-size: 1.65rem; font-weight: 750; margin-top: .15rem;}
    .metric-detail {font-size: .78rem; color: #94a3b8; margin-top: .15rem;}
    .model-box {
        padding: 1rem 1.1rem;
        border-radius: 14px;
        border: 1px solid #334155;
        background: rgba(15, 23, 42, .55);
        height: 100%;
    }
    .model-box h4 {margin: 0 0 .35rem 0;}
    .model-box p {margin: .25rem 0; color: #cbd5e1; font-size: .9rem;}
    .pill {
        display: inline-block;
        padding: .18rem .55rem;
        border-radius: 999px;
        background: #1e3a8a;
        color: #bfdbfe;
        font-size: .72rem;
        font-weight: 650;
        margin-bottom: .45rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

RES = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------
# Experiment data
# ---------------------------------------------------------------------
COLS = [
    "Model", "Type", "R@1", "R@5", "R@10", "Mean rank",
    "Paper R@1", "Paper R@10", "Paper mean rank", "Log"
]

ROWS = [
    ("Baseline", "Baseline", 9.3, None, 45.9, 37.4, 10.3, 17.1, 176.1, None),
    ("Baseline + Weight", "Baseline", 11.2, None, 51.5, 26.6, 19.8, 65.5, 10.5, None),
    ("GRU (lr 1e-4)", "Main", 36.5, 72.2, 85.2, 7.6, 37.0, 86.8, 7.3, "results_gru_lr1e-4.txt"),
    ("LSTM (lr 1e-4)", "Main", 36.5, 72.5, 85.4, 7.6, 35.4, 86.2, 7.5, "results_lstm_lr1e-4.txt"),
    ("Transformer (lr 1e-4)", "Ours", 33.7, 69.9, 83.3, 8.4, None, None, None, "results_transformer_glove.txt"),
    ("GRU (lr 1e-3)", "Ablation", 30.6, 65.9, 80.3, 9.7, None, None, None, "results_gru_glove.txt"),
    ("LSTM (lr 1e-3)", "Ablation", 30.4, 66.2, 80.5, 10.1, None, None, None, "results_lstm_glove.txt"),
]

df = pd.DataFrame(ROWS, columns=COLS)

PAPER_NAME = "Qian & Lamberti — Neural Caption-Image Retrieval"
PAPER_URL = "https://cs229.stanford.edu/proj2018/report/59.pdf"

MODEL_ORDER = [
    "Baseline",
    "Baseline + Weight",
    "GRU (lr 1e-4)",
    "LSTM (lr 1e-4)",
    "Transformer (lr 1e-4)",
    "GRU (lr 1e-3)",
    "LSTM (lr 1e-3)",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
LOG_RE = re.compile(
    r"epoch\s+(\d+)\s+loss\s+([\d.]+)\s+\|\s+"
    r"val R@1\s+([\d.]+)\s+R@5\s+([\d.]+)\s+"
    r"R@10\s+([\d.]+)\s+mean_r\s+([\d.]+)"
)

LOG_ALIASES = {
    "GRU (lr 1e-4)": ["results_gru_glove.txt"],
    "LSTM (lr 1e-4)": ["results_lstm_glove.txt"],
    "Transformer (lr 1e-4)": ["results_transformer_glove.txt"],
    "GRU (lr 1e-3)": [],
    "LSTM (lr 1e-3)": [],
}

@st.cache_data
def load_log(model_name):
    """Load a training log if it exists locally."""
    for filename in LOG_ALIASES.get(model_name, []):
        path = RES / filename
        if path.exists():
            rows = []
            with path.open(encoding="utf-8", errors="ignore") as f:
                for line in f:
                    match = LOG_RE.search(line)
                    if match:
                        rows.append([float(x) for x in match.groups()])
            if rows:
                return pd.DataFrame(
                    rows,
                    columns=["epoch", "loss", "R@1", "R@5", "R@10", "Mean rank"],
                )
    return None


def model_group(name):
    if name.startswith("GRU"):
        return "GRU"
    if name.startswith("LSTM"):
        return "LSTM"
    if name.startswith("Transformer"):
        return "Transformer"
    if name.startswith("Baseline +"):
        return "Baseline + Weight"
    return "Baseline"


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
st.sidebar.title("Dashboard controls")
st.sidebar.caption("Neural Caption–Image Retrieval")

show_ablations = st.sidebar.checkbox("Show lr = 1e-3 ablations", value=True)

if show_ablations:
    visible_df = df.copy()
else:
    visible_df = df[df["Type"] != "Ablation"].copy()

st.sidebar.divider()
st.sidebar.markdown("**Evaluation setup**")
st.sidebar.write("MS COCO")
st.sidebar.write("1,000-image retrieval test set")
st.sidebar.write("Single run · seed 0")
st.sidebar.write("Shared embedding: 1,024-D")

st.sidebar.divider()
st.sidebar.link_button("Open reference paper", PAPER_URL)


# ---------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">ML mini-project · retrieval</div>
        <h1>Caption → Image Retrieval</h1>
        <p>
            Can a sentence encoder place a caption close to its matching image
            in a shared 1,024-dimensional space?
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Replication of the Stanford CS229 project by Qian & Lamberti, "
    "with a Transformer caption encoder added as our extension."
)


# ---------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------
tab_overview, tab_results, tab_compare, tab_training = st.tabs(
    ["Project overview", "Results", "Compare models", "Training dynamics"]
)


with tab_overview:
    st.subheader("The problem we are solving")
    st.markdown(
        """
        The system performs **caption-to-image retrieval**. A natural-language
        query is encoded into the same shared space as precomputed image
        features. The nearest image embeddings are treated as the retrieved
        results.

        The reference project studies recurrent sentence encoders such as
        **GRU and LSTM**. We reproduced that setup and then added a
        **Transformer caption encoder** as our own extension.
        """
    )

    st.markdown("### What the experiment compares")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="model-box">
                <span class="pill">REFERENCE</span>
                <h4>GRU / LSTM</h4>
                <p>Learned recurrent caption encoders trained to align captions with image features.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="model-box">
                <span class="pill">BASELINES</span>
                <h4>Average GloVe</h4>
                <p>Non-recurrent baselines using averaged word embeddings, including the weighted variant.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="model-box">
                <span class="pill">OUR EXTENSION</span>
                <h4>Transformer</h4>
                <p>2-layer self-attention encoder with learned positions and mean pooling.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Headline results")

    best_r1 = df.loc[df["R@1"].idxmax()]
    best_r10 = df.loc[df["R@10"].idxmax()]
    best_rank = df.loc[df["Mean rank"].idxmin()]
    transformer = df[df["Model"].str.startswith("Transformer")].iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    cards = [
        ("Best R@1", f"{best_r1['R@1']:.1f}%", best_r1["Model"]),
        ("Best R@10", f"{best_r10['R@10']:.1f}%", best_r10["Model"]),
        ("Best mean rank", f"{best_rank['Mean rank']:.1f}", best_rank["Model"]),
        ("Transformer R@1", f"{transformer['R@1']:.1f}%", "our extension"),
    ]

    for col, (label, value, detail) in zip([m1, m2, m3, m4], cards):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-detail">{detail}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### What we learned")
    st.markdown(
        """
        - **GRU and LSTM are essentially tied** in our main runs.
        - Lowering the learning rate from **1e-3 to 1e-4** produced a large improvement.
        - The **Transformer did not beat the recurrent encoders** at the matched learning rate.
        - The Transformer is an extension of the reference setup, not a model reported by the paper.
        """
    )


# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------
with tab_results:
    st.subheader("Reported vs. obtained results")
    st.markdown(
        "Test-set retrieval on 1,000 images. Paper values are from Table 1; "
        "our values are the single seed-0 runs."
    )

    result_view = visible_df[
        ["Model", "Type", "R@1", "R@5", "R@10", "Mean rank"]
    ].copy()

    st.dataframe(
        result_view.style.format(
            {
                "R@1": lambda x: "—" if pd.isna(x) else f"{x:.1f}",
                "R@5": lambda x: "—" if pd.isna(x) else f"{x:.1f}",
                "R@10": lambda x: "—" if pd.isna(x) else f"{x:.1f}",
                "Mean rank": lambda x: "—" if pd.isna(x) else f"{x:.1f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Paper replication at a glance")

    paper_df = df[df["Paper R@10"].notna()].copy()
    paper_long = pd.DataFrame(
        {
            "Model": list(paper_df["Model"]),
            "Paper": list(paper_df["Paper R@10"]),
            "Ours": list(paper_df["R@10"]),
        }
    ).melt(id_vars="Model", var_name="Source", value_name="R@10")

    fig = px.bar(
        paper_long,
        x="Model",
        y="R@10",
        color="Source",
        barmode="group",
        text="R@10",
        template="plotly_dark",
        title="R@10 — paper vs. our replication",
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(
        yaxis_title="Recall@10 (%)",
        xaxis_title=None,
        legend_title=None,
        height=430,
        margin=dict(l=20, r=20, t=65, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------
# Compare models — intentionally kept as the main comparison view
# ---------------------------------------------------------------------
with tab_compare:
    st.subheader("Compare models")
    st.markdown("Use the controls to inspect the metric that matters for retrieval.")

    metric = st.selectbox(
        "Metric",
        ["R@1", "R@5", "R@10", "Mean rank"],
        index=2,
    )

    compare_df = visible_df[["Model", "Type", metric]].dropna().copy()
    compare_df["Model"] = pd.Categorical(
        compare_df["Model"], categories=MODEL_ORDER, ordered=True
    )
    compare_df = compare_df.sort_values("Model")

    fig = px.bar(
        compare_df,
        x="Model",
        y=metric,
        color="Type",
        text=metric,
        template="plotly_dark",
        title=f"{metric} across our models",
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(
        height=500,
        xaxis_title=None,
        yaxis_title=f"{metric}" + (" (%)" if metric != "Mean rank" else ""),
        legend_title=None,
        margin=dict(l=20, r=20, t=65, b=20),
    )
    if metric == "Mean rank":
        fig.update_yaxes(type="log", title="Mean rank · lower is better")
    else:
        fig.update_yaxes(range=[0, 100], title=f"{metric} (%)")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Main-model trade-off")

    neural = visible_df[
        visible_df["Model"].isin(
            ["GRU (lr 1e-4)", "LSTM (lr 1e-4)", "Transformer (lr 1e-4)"]
        )
    ][["Model", "R@1", "R@5", "R@10", "Mean rank"]].copy()

    st.dataframe(
        neural.style.format("{:.1f}", subset=["R@1", "R@5", "R@10", "Mean rank"]),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------
# Training dynamics
# ---------------------------------------------------------------------
with tab_training:
    st.subheader("Training dynamics")
    st.markdown(
        "Explore validation behaviour epoch by epoch. "
        "The selected epoch corresponds to the best validation R@10 used for early stopping."
    )

    available = []
    missing = []
    for model in LOG_ALIASES:
        if load_log(model) is not None:
            available.append(model)
        else:
            missing.append(model)

    if not available:
        st.info(
            "No matching training-log files were found in results/. "
            "The dashboard can still show all final test-set results. "
            "Add the *.txt logs to results/ to enable these interactive curves."
        )
    else:
        selected = st.multiselect(
            "Models",
            available,
            default=[m for m in available if "lr 1e-4" in m],
        )

        metric = st.selectbox(
            "Training plot",
            ["R@10", "R@1", "Mean rank", "loss"],
            format_func=lambda x: {
                "R@10": "Validation R@10",
                "R@1": "Validation R@1",
                "Mean rank": "Validation mean rank",
                "loss": "Training loss",
            }[x],
        )

        if selected:
            fig = go.Figure()

            for model in selected:
                log = load_log(model)
                if log is None:
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=log["epoch"],
                        y=log[metric],
                        mode="lines+markers",
                        name=model,
                        hovertemplate=(
                            "<b>%{fullData.name}</b><br>"
                            "Epoch %{x}<br>"
                            f"{metric}: %{{y:.2f}}<extra></extra>"
                        ),
                    )
                )

                # Mark the selected/best epoch by validation R@10.
                best_idx = log["R@10"].idxmax()
                best_row = log.loc[best_idx]
                fig.add_trace(
                    go.Scatter(
                        x=[best_row["epoch"]],
                        y=[best_row[metric]],
                        mode="markers",
                        name=f"{model} · selected epoch",
                        showlegend=False,
                        marker=dict(size=11, symbol="diamond"),
                        hovertemplate=(
                            f"<b>{model}</b><br>"
                            f"Selected epoch: {int(best_row['epoch'])}<br>"
                            f"Validation R@10: {best_row['R@10']:.2f}<extra></extra>"
                        ),
                    )
                )

            fig.update_layout(
                template="plotly_dark",
                height=520,
                hovermode="x unified",
                xaxis_title="Epoch",
                yaxis_title={
                    "R@10": "Validation R@10 (%)",
                    "R@1": "Validation R@1 (%)",
                    "Mean rank": "Validation mean rank · lower is better",
                    "loss": "Training loss",
                }[metric],
                legend_title=None,
                margin=dict(l=20, r=20, t=35, b=20),
            )

            if metric in {"R@10", "R@1"}:
                fig.update_yaxes(range=[0, 100])

            st.plotly_chart(fig, use_container_width=True)

            # Compact run summaries
            st.markdown("### Run summaries")
            summaries = []
            for model in selected:
                log = load_log(model)
                if log is None:
                    continue
                best_idx = log["R@10"].idxmax()
                best = log.loc[best_idx]
                summaries.append(
                    {
                        "Model": model,
                        "Epochs trained": int(log["epoch"].max()),
                        "Best epoch": int(best["epoch"]),
                        "Best val R@10": best["R@10"],
                        "Final loss": log.iloc[-1]["loss"],
                    }
                )

            if summaries:
                st.dataframe(
                    pd.DataFrame(summaries).style.format(
                        {
                            "Best val R@10": "{:.1f}",
                            "Final loss": "{:.3f}",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        if missing:
            with st.expander("Logs not available locally"):
                st.write(
                    "The following runs have no matching log file in results/: "
                    + ", ".join(missing)
                )


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------
st.divider()
st.caption(
    f"Reference: {PAPER_NAME}. "
    "Our Transformer is an extension evaluated under the same retrieval setup; "
    "it is not part of the reference paper."
)
