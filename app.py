"""Offline Streamlit demonstration for the cybersecurity world model."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import altair as alt
import pandas as pd
import streamlit as st

from cyberworld.config import load_config
from cyberworld.data.normalization import StandardNormalizer
from cyberworld.demo import (
    BUNDLED_SAMPLES,
    FEATURED_SAMPLE_NAMES,
    analyze_trajectory,
    benchmark_views,
    explain_peak,
    feature_availability,
    feature_matrix,
    load_benchmark,
    load_bundled_sample,
    load_uploaded_input,
)
from cyberworld.explain import load_model_from_checkpoint
from cyberworld.runtime import select_device

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/experiment/full_combined.yaml"
CHECKPOINT_PATH = ROOT / "checkpoints/world_model_full_combined_lstm/best.pt"
NORMALIZER_PATH = ROOT / "checkpoints/world_model_full_combined_lstm/normalizer.npz"
BENCHMARK_PATH = ROOT / "runs/world_model_full_combined_lstm/benchmark.json"
THRESHOLD = 0.2


@st.cache_resource
def resources():
    config = load_config(CONFIG_PATH, repo_root=ROOT)
    model, checkpoint = load_model_from_checkpoint(CHECKPOINT_PATH)
    normalizer = StandardNormalizer.load(NORMALIZER_PATH)
    device = select_device(config.training.device)
    return config, model, checkpoint, normalizer, device


def stage_style(value: str) -> str:
    colors = {
        "NORMAL": "background-color: #e8f5e9",
        "INITIAL_ACCESS": "background-color: #fff3cd",
        "IMPACT": "background-color: #f8d7da",
        "OTHER_MALICIOUS": "background-color: #fde2e2",
    }
    return colors.get(value, "background-color: #e8eef7")


def timeline_chart(frame: pd.DataFrame) -> alt.Chart:
    base = alt.Chart(frame).encode(x=alt.X("timestamp:T", title="Time"))
    flagged = (
        base.transform_filter(alt.datum.above_threshold)
        .mark_area(color="#ef4444", opacity=0.16)
        .encode(y=alt.Y("risk_score:Q", scale=alt.Scale(domain=[0, 1]), title="Risk probability"))
    )
    line = base.mark_line(color="#2563eb", strokeWidth=2).encode(
        y=alt.Y("risk_score:Q", scale=alt.Scale(domain=[0, 1]), title="Risk probability"),
        tooltip=[
            "window_index:Q",
            "timestamp:T",
            alt.Tooltip("risk_score:Q", format=".4f"),
            "predicted_stage:N",
        ],
    )
    threshold = base.mark_line(color="#dc2626", strokeDash=[6, 4]).encode(y="threshold:Q")
    return (flagged + line + threshold).properties(height=320)


def metric_table(frame: pd.DataFrame):
    display = frame.copy()
    for column in ("precision", "recall", "f1", "fpr"):
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    display["threshold"] = display["threshold"].map(lambda value: f"{value:.1f}")
    st.dataframe(display, hide_index=True, use_container_width=True)


def load_prepared_selection() -> None:
    name = st.session_state.prepared_sample
    st.session_state.loaded = load_bundled_sample(name, config)
    st.session_state.pop("analysis", None)
    st.session_state.pop("attribution", None)


st.set_page_config(page_title="CyberWorld — Predictive Defence", page_icon="🛡️", layout="wide")
config, model, checkpoint, normalizer, device = resources()

st.title("CyberWorld: Predictive Cyber Defence")
st.caption(
    "Offline network-state forecasting, attack-stage prediction, risk scoring, and explainability"
)

with st.sidebar:
    st.header("Navigation")
    for index, title in enumerate(
        (
            "Input & Feature Extraction",
            "World Model & Inference",
            "Infiltration Prediction",
            "Flagged Flows",
            "Explainability",
            "Benchmark",
            "Model & Data Summary",
        ),
        start=1,
    ):
        st.markdown(f"{index}. {title}")
    st.divider()
    st.caption("Runs fully offline. No external APIs or telemetry.")

st.header("1. Input & Feature Extraction")
left, right = st.columns([1, 1])
with left:
    st.subheader("Bundled demonstrations")
    sample_columns = st.columns(2)
    for column, name in zip(sample_columns, FEATURED_SAMPLE_NAMES, strict=True):
        if column.button(name, use_container_width=True):
            st.session_state.loaded = load_bundled_sample(name, config)
            st.session_state.pop("analysis", None)
            st.session_state.pop("attribution", None)
    selected_sample = st.selectbox(
        "Prepared trajectory library",
        tuple(BUNDLED_SAMPLES),
        key="prepared_sample",
        on_change=load_prepared_selection,
        help="Switch among all 11 prepared trajectories without uploading source files.",
    )
    st.button(
        "Load selected trajectory",
        on_click=load_prepared_selection,
        use_container_width=True,
    )
with right:
    st.subheader("Upload traffic evidence")
    uploaded = st.file_uploader("PCAP or supported flow CSV", type=["pcap", "pcapng", "csv"])
    dataset_id = st.selectbox(
        "CSV schema",
        ("cic_ids2018", "ctu13", "dapt2020", "unsw_nb15"),
        help="Ignored for PCAP uploads.",
    )
    if uploaded and st.button("Extract uploaded traffic", type="primary"):
        with (
            st.spinner("Extracting canonical flows and 10-second feature windows…"),
            TemporaryDirectory(prefix="cyberworld-upload-") as directory,
        ):
            path = Path(directory) / uploaded.name
            path.write_bytes(uploaded.getvalue())
            st.session_state.loaded = load_uploaded_input(
                path, dataset_id=dataset_id, config=config
            )
        st.session_state.pop("analysis", None)
        st.session_state.pop("attribution", None)

loaded = st.session_state.get("loaded")
if loaded is None:
    st.info("Choose a bundled sample or upload a PCAP/CSV to begin.")
    st.stop()

availability = feature_availability(loaded)
flow_available = int(availability.loc[availability["family"] == "flow-level", "populated"].sum())
packet_available = int(
    availability.loc[availability["family"] == "packet-derived", "populated"].sum()
)
summary_columns = st.columns(4)
summary_columns[0].metric("Extracted windows", f"{len(loaded.trajectory.states):,}")
summary_columns[1].metric("Canonical features", len(availability))
summary_columns[2].metric("Flow-level populated", flow_available)
summary_columns[3].metric("Packet-derived populated", packet_available)
st.caption(
    f"{loaded.source_kind} · {loaded.origin.isoformat()} → {loaded.end.isoformat()} · "
    "Unavailable source fields are explicitly masked rather than filled with invented values. "
    "PCAP inputs populate TTL, TCP-window variance, fragments, retransmissions, payload, and "
    "port-scan signatures through the packet extractor."
)
with st.expander("Inspect the actual 51-feature matrix and availability", expanded=False):
    st.dataframe(feature_matrix(loaded).head(12), hide_index=True, use_container_width=True)
    st.dataframe(availability, hide_index=True, use_container_width=True)

st.header("2. World Model & Inference")
model_columns = st.columns(4)
model_columns[0].metric("Architecture", "Probabilistic LSTM")
model_columns[1].metric("Checkpoint epoch", int(checkpoint["epoch"]))
model_columns[2].metric("History", f"{config.data.history_windows} windows")
model_columns[3].metric("Forecast horizon", f"{config.data.forecast_windows} windows")
st.write(
    "This is a state-transition model learning **P(Sₜ₊₁ | Sₜ)** through autoregressive "
    "rollout—not a static per-flow classifier."
)
if st.button("Run Analysis", type="primary", use_container_width=True):
    with st.spinner("Running the six-step world-model rollout and integrated gradients…"):
        analysis = analyze_trajectory(
            loaded,
            model=model,
            normalizer=normalizer,
            config=config,
            device=device.device,
            threshold=THRESHOLD,
        )
        attribution = explain_peak(
            loaded,
            analysis,
            model=model,
            normalizer=normalizer,
            config=config,
        )
        st.session_state.analysis = analysis
        st.session_state.attribution = attribution

analysis = st.session_state.get("analysis")
if analysis is None:
    st.info(
        "Run the analysis to populate prediction, annotation, explanation, and benchmark sections."
    )
    st.stop()

st.header("3. Infiltration Prediction Engine")
st.caption(
    f"K-step forward simulation with K={config.data.forecast_windows}; "
    f"deployed operating threshold={THRESHOLD:.1f}."
)
st.altair_chart(timeline_chart(analysis.timeline), use_container_width=True)
peak_columns = st.columns(3)
peak_columns[0].metric("Peak window", analysis.peak_window_index)
peak_columns[1].metric("Peak malicious-risk", f"{analysis.peak_score:.4f}")
peak_columns[2].metric("Predicted attack stage", analysis.peak_stage.name)
st.subheader("Peak context: complete K-step predicted trajectory")
st.line_chart(analysis.peak_horizon.set_index("forecast_step")["risk_score"], height=220)
st.dataframe(analysis.peak_horizon, hide_index=True, use_container_width=True)

st.header("4. Flagged Flows & Attack Stage Annotations")
st.caption(
    "Each row is a flagged 10-second network-state window. Context values are raw aggregated "
    "features from that window, not normalized model inputs."
)
flagged_columns = [
    "window_index",
    "timestamp",
    "risk_score",
    "predicted_stage",
    "protocol",
    "unique_dst_ports",
    "mean_duration",
    "total_flows",
]
flagged = analysis.flagged[flagged_columns]
st.metric("Windows above 0.2", f"{len(flagged):,}")
st.dataframe(
    flagged.style.map(stage_style, subset=["predicted_stage"]).format({"risk_score": "{:.4f}"}),
    hide_index=True,
    use_container_width=True,
    height=min(600, 38 * (len(flagged) + 1)),
)

st.header("5. Explainability")
attribution = st.session_state.attribution
chart = (
    alt.Chart(attribution)
    .mark_bar()
    .encode(
        x=alt.X("attribution:Q", title="Signed integrated-gradient attribution"),
        y=alt.Y("feature:N", sort="-x", title=None),
        color=alt.condition("datum.attribution >= 0", alt.value("#dc2626"), alt.value("#2563eb")),
        tooltip=[
            "feature:N",
            alt.Tooltip("attribution:Q", format=".5f"),
            alt.Tooltip("raw_value:Q", format=".4f"),
        ],
    )
    .properties(height=340)
)
st.altair_chart(chart, use_container_width=True)
st.dataframe(
    attribution[["feature", "attribution", "raw_value", "available"]],
    hide_index=True,
    use_container_width=True,
)
st.caption(
    "Feature attribution via integrated gradients, computed in the model's normalized input "
    "space and reported against real feature values"
)

st.header("6. Benchmark — World Model vs. Logistic Regression")
report = load_benchmark(BENCHMARK_PATH, loaded.trajectory.trajectory_id)
if report is None:
    st.info(
        "This uploaded trajectory is not one of the four audited held-out benchmark trajectories. "
        "No validation metric is synthesized for it."
    )
else:
    fixed, observed_best = benchmark_views(report)
    st.subheader("Fixed threshold 0.5 — directly comparable operating point")
    metric_table(fixed)
    st.subheader("Best observed threshold — descriptive validation sweep")
    metric_table(observed_best)
    visual = pd.concat(
        [
            fixed.assign(view="Fixed 0.5"),
            observed_best.assign(view="Best observed"),
        ]
    ).melt(
        id_vars=["model", "view", "threshold"],
        value_vars=["f1", "fpr"],
        var_name="metric",
        value_name="value",
    )
    comparison_chart = (
        alt.Chart(visual)
        .mark_bar()
        .encode(
            x=alt.X("model:N", title=None),
            y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("model:N", legend=None),
            column=alt.Column("metric:N", title=None),
            row=alt.Row("view:N", title=None),
            tooltip=["model:N", "view:N", "metric:N", alt.Tooltip("value:Q", format=".4f")],
        )
        .properties(height=150)
    )
    st.altair_chart(comparison_chart, use_container_width=True)
st.write(
    "The probabilistic LSTM provides better precision/FPR tradeoffs and higher best observed "
    "F1 across all four held-out attack families, though its raw probabilities are not fully "
    "calibrated across datasets at a fixed threshold."
)

st.header("7. Model & Data Summary")
summary_left, summary_right = st.columns(2)
with summary_left:
    st.subheader("Training evidence")
    st.markdown(
        "- 11 real trajectories across 7 attack families\n"
        "- CIC-IDS2018 and CTU-13 sources\n"
        "- Whole-trajectory split: 7 train / 4 validation\n"
        "- No flow or forecast window crosses a trajectory boundary"
    )
with summary_right:
    st.subheader("Architecture")
    st.markdown(
        "- 51-dimensional masked network state\n"
        "- Probabilistic autoregressive LSTM\n"
        "- State, attack-stage, malicious-risk, and compromise-risk heads\n"
        "- Integrated-gradient explanation over the normalized temporal input"
    )
st.markdown("Full experimental write-up: `docs/MODEL_RESULTS.md`")
