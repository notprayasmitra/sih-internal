"""Flask presentation layer for the offline CyberWorld demonstration."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from flask import Flask, redirect, render_template, request, url_for

from cyberworld.config import load_config
from cyberworld.data.normalization import StandardNormalizer
from cyberworld.demo import (
    BUNDLED_SAMPLES,
    FEATURED_SAMPLE_NAMES,
    DemoInput,
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

app = Flask(__name__)
app.config.update(MAX_CONTENT_LENGTH=512 * 1024 * 1024)

config = load_config(CONFIG_PATH, repo_root=ROOT)
model, checkpoint = load_model_from_checkpoint(CHECKPOINT_PATH)
normalizer = StandardNormalizer.load(NORMALIZER_PATH)
device = select_device(config.training.device)


def _records(frame: Any) -> list[dict[str, Any]]:
    return frame.to_dict(orient="records")


def build_page(loaded: DemoInput, *, run_analysis: bool) -> dict[str, Any]:
    availability = feature_availability(loaded)
    matrix = feature_matrix(loaded)
    context: dict[str, Any] = {
        "samples": tuple(BUNDLED_SAMPLES),
        "featured_samples": FEATURED_SAMPLE_NAMES,
        "selected_sample": loaded.name,
        "loaded": loaded,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "window_count": len(loaded.trajectory.states),
        "feature_count": len(availability),
        "flow_available": int(
            availability.loc[availability["family"] == "flow-level", "populated"].sum()
        ),
        "packet_available": int(
            availability.loc[availability["family"] == "packet-derived", "populated"].sum()
        ),
        "matrix_columns": list(matrix.columns),
        "matrix_rows": _records(matrix.head(8)),
        "availability_rows": _records(availability),
        "analysis": None,
    }
    if not run_analysis:
        return context

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
    benchmark = load_benchmark(BENCHMARK_PATH, loaded.trajectory.trajectory_id)
    fixed_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    if benchmark is not None:
        fixed, observed_best = benchmark_views(benchmark)
        fixed_rows = _records(fixed)
        best_rows = _records(observed_best)
    context["analysis"] = {
        "timeline": _records(
            analysis.timeline[["window_index", "timestamp", "risk_score", "predicted_stage"]]
        ),
        "peak_window": analysis.peak_window_index,
        "peak_score": analysis.peak_score,
        "peak_stage": analysis.peak_stage.name,
        "horizon": _records(analysis.peak_horizon),
        "flagged_count": len(analysis.flagged),
        "flagged": _records(analysis.flagged.head(250)),
        "attribution": _records(attribution),
        "benchmark_fixed": fixed_rows,
        "benchmark_best": best_rows,
    }
    return context


@app.get("/")
def index():
    sample = request.args.get("sample", FEATURED_SAMPLE_NAMES[0])
    if sample not in BUNDLED_SAMPLES:
        sample = FEATURED_SAMPLE_NAMES[0]
    loaded = load_bundled_sample(sample, config)
    return render_template(
        "index.html",
        **build_page(loaded, run_analysis=request.args.get("analyze") == "1"),
    )


@app.get("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html")


@app.post("/upload")
def upload():
    uploaded = request.files.get("traffic_file")
    if uploaded is None or not uploaded.filename:
        return index()
    dataset_id = request.form.get("dataset_id", "cic_ids2018")
    with TemporaryDirectory(prefix="cyberworld-flask-upload-") as directory:
        path = Path(directory) / Path(uploaded.filename).name
        uploaded.save(path)
        loaded = load_uploaded_input(path, dataset_id=dataset_id, config=config)
    return render_template("index.html", **build_page(loaded, run_analysis=True))


@app.get("/upload")
def upload_get():
    """Keep accidental navigation to the upload endpoint user-friendly."""
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
