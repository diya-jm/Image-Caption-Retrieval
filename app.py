"""Results dashboard for the Neural Caption-Image Retrieval mini-project.
Run from the repo root:  streamlit run app.py
Reads training logs from results/ and (optionally) figures from results/figures/.
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Neural Caption-Image Retrieval", layout="wide")
RES = Path(__file__).parent / "results"

# ---------------------------------------------------------------- data
# Test set, 1K images, single run (seed 0). Paper values = Table 1 of the reference paper.
COLS = ["Model", "Type", "R@1", "R@5", "R@10", "Mean rank",
        "Paper R@1", "Paper R@10", "Paper mean rank", "Log"]
ROWS = [
    ("Baseline",               "Baseline", 9.3,  None, 45.9, 37.4, 10.3, 17.1, 176.1, None),
    ("Baseline + Weight",      "Baseline", 11.2, None, 51.5, 26.6, 19.8, 65.5, 10.5,  None),
    ("GRU (lr 1e-4)",          "Main",     36.5, 72.2, 85.2, 7.6,  37.0, 86.8, 7.3,   "results_gru_lr1e-4.txt"),
    ("LSTM (lr 1e-4)",         "Main",     36.5, 72.5, 85.4, 7.6,  35.4, 86.2, 7.5,   "results_lstm_lr1e-4.txt"),
    ("Transformer (lr 1e-4)",  "Ours",     33.7, 69.9, 83.3, 8.4,  None, None, None,  "results_transformer_glove.txt"),
    ("GRU (lr 1e-3)",          "Ablation", 30.6, 65.9, 80.3, 9.7,  None, None, None,  "results_gru_glove.txt"),
    ("LSTM (lr 1e-3)",         "Ablation", 30.4, 66.2, 80.5, 10.1, None, None, None,  "results_lstm_glove.txt"),
]
df = pd.DataFrame(ROWS, columns=COLS)
COLOR = {"GRU": "#0072B2", "LSTM": "#E69F00", "Transformer": "#009E73",
         "Baseline +": "#56B4E9", "Baseline": "#999999"}


def color_of(name):
    for key, c in COLOR.items():
        if name.startswith(key):
            return c
    return "#999999"


LOG_RE = re.compile(r"epoch\s+(\d+) loss ([\d.]+) \| val R@1 ([\d.]+) R@5 ([\d.]+) R@10 ([\d.]+) mean_r ([\d.]+)")


@st.cache_data
def load_log(fname):
    path = RES / fname
    if not path.exists():
        return None
    rows = [list(map(float, m.groups())) for m in map(LOG_RE.search, open(path)) if m]
    if not rows:
        return None
    return pd.DataFrame(rows, columns=["epoch", "loss", "R@1", "R@5", "R@10", "Mean rank"]).set_index("epoch")


# ---------------------------------------------------------------- sidebar
st.sidebar.header("Options")
show_abl = st.sidebar.checkbox("Show lr 1e-3 ablations", value=True)
view = df if show_abl else df[df["Type"] != "Ablation"]

# ---------------------------------------------------------------- header
st.title("Neural Caption-Image Retrieval")
st.caption("UE24CS352A Machine Learning Mini-Project | Chirag Arun Yadwad & Diya J Marar | MS COCO, 1,000-image retrieval")
st.write("Type a sentence, get the matching photos. Images and captions are mapped into a shared 1,024-d space; "
         "the nearest images to a caption are returned.")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Results table", "Compare models", "Training curves"])

# ---------------------------------------------------------------- overview
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best R@1", "36.5%", "GRU = LSTM (lr 1e-4)", delta_color="off")
    c2.metric("Best R@10", "85.4%", "LSTM (lr 1e-4)", delta_color="off")
    c3.metric("Best mean rank", "7.6", "GRU = LSTM (lr 1e-4)", delta_color="off")
    c4.metric("Transformer (ours) R@1", "33.7%", "-2.8 vs GRU/LSTM")
    st.subheader("Key findings")
    st.markdown(
        "- Learned recurrent encoders (GRU/LSTM) **far outperform** the averaged-GloVe baselines, as in the paper.\n"
        "- **GRU is about equal to LSTM** (R@1 36.5 vs 36.5, R@10 85.2 vs 85.4).\n"
        "- At lr 1e-4 we are within **~0.5 R@1 and 1.6 R@10** of the paper's GRU.\n"
        "- **Learning rate was the main cause of our initial gap:** lr 1e-3 -> 1e-4 gave about +6 R@1 and +5 R@10.\n"
        "- Our **Transformer extension did not beat GRU/LSTM** at the matched lr (R@1 33.7 vs 36.5) and is slower to train.\n"
        "- Limitations: single seed, untuned hyper-parameters, fixed VGG19 features."
    )
    with st.expander("What do the metrics mean?"):
        st.markdown("**R@K**: % of caption queries whose correct image is in the top K results. "
                    "**Mean rank**: average position of the correct image out of 1,000 (lower is better).")

# ---------------------------------------------------------------- table
with tab2:
    st.dataframe(
        view.drop(columns=["Log"]).set_index("Model"),
        use_container_width=True,
    )
    st.caption("Paper values are from Table 1 of Qian & Lamberti. 'Ours' are single runs (seed 0). "
               "R@5 was not recorded for the baselines.")

# ---------------------------------------------------------------- compare
with tab3:
    metric = st.selectbox("Metric", ["R@1", "R@5", "R@10", "Mean rank"], index=2)
    d = view.dropna(subset=[metric]).sort_values(metric, ascending=(metric == "Mean rank"))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ax.barh(d["Model"], d[metric], color=[color_of(n) for n in d["Model"]])
    for b, t in zip(bars, d["Type"]):
        if t == "Ablation":
            b.set_alpha(.45); b.set_hatch("//")
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {b.get_width():.1f}", va="center", fontsize=9)
    if metric == "Mean rank":
        ax.set_xscale("log"); ax.set_xlabel("Mean rank (log scale, lower is better)")
    else:
        ax.set_xlim(0, 100); ax.set_xlabel(f"Test {metric} (%)")
    ax.invert_yaxis(); ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)

    figs = RES / "figures"
    if figs.exists():
        with st.expander("Saved figures from the repo"):
            for p in sorted(figs.glob("*.png")):
                st.image(str(p), caption=p.name)

# ---------------------------------------------------------------- curves
with tab4:
    runs = [(m, f) for m, f in zip(view["Model"], view["Log"]) if f]
    pick = st.multiselect("Runs", [m for m, _ in runs], default=[m for m, _ in runs if "1e-4" in m])
    what = st.selectbox("Plot", ["R@10", "R@1", "Mean rank", "loss"], index=0,
                        format_func=lambda x: {"R@10": "Validation R@10", "R@1": "Validation R@1",
                                               "Mean rank": "Validation mean rank", "loss": "Training loss"}[x])
    series = {}
    for m, f in runs:
        if m in pick:
            log = load_log(f)
            if log is None:
                st.warning(f"Log not found: results/{f}")
            else:
                series[m] = log[what]
    if series:
        st.line_chart(pd.DataFrame(series))