"""Command-line entry points for data preparation, diagnostics, and training."""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from pathlib import Path
from typing import cast

import numpy as np
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from cyberworld.adapters import read_flow_csv
from cyberworld.baseline import compute_metrics, make_baseline_samples, train_logistic_baseline
from cyberworld.config import AppConfig, load_config
from cyberworld.data.catalog import discover_files, load_catalog
from cyberworld.data.normalization import StandardNormalizer
from cyberworld.data.synthetic import make_synthetic_trajectories
from cyberworld.data.trajectory import Trajectory, TrajectoryWindowDataset, load_trajectories
from cyberworld.explain import attribute_prediction, load_model_from_checkpoint
from cyberworld.models import (
    TemporalConvolutionalWorldModel,
    TemporalJEPAWorldModel,
    TemporalTransformerWorldModel,
    WorldModel,
)
from cyberworld.pcap import extract_pcap
from cyberworld.preprocessing import STATE_FEATURES, build_trajectory
from cyberworld.runtime import seed_everything, select_device
from cyberworld.training import Trainer
from cyberworld.training.losses import inverse_frequency_stage_weights


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load(path: str) -> AppConfig:
    root = _repo_root()
    load_dotenv(root / ".env")
    return load_config(path, repo_root=root)


def _split_trajectories(
    trajectories: list[Trajectory], validation_fraction: float, seed: int
) -> tuple[list[Trajectory], list[Trajectory]]:
    if len(trajectories) < 2:
        raise ValueError("At least two independent trajectories are required for train/validation")
    shuffled = list(trajectories)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_fraction))
    validation_count = min(validation_count, len(shuffled) - 1)
    return shuffled[validation_count:], shuffled[:validation_count]


def _split_trajectories_explicit(
    trajectories: list[Trajectory],
    train_trajectory_ids: tuple[str, ...],
    validation_trajectory_ids: tuple[str, ...],
) -> tuple[list[Trajectory], list[Trajectory]]:
    by_id = {trajectory.trajectory_id: trajectory for trajectory in trajectories}
    if len(by_id) != len(trajectories):
        raise ValueError("Loaded trajectory IDs must be unique")
    configured_ids = set(train_trajectory_ids) | set(validation_trajectory_ids)
    missing = configured_ids - set(by_id)
    if missing:
        raise ValueError(f"Configured trajectory IDs were not found: {sorted(missing)}")
    unassigned = set(by_id) - configured_ids
    if unassigned:
        raise ValueError(
            f"Loaded trajectories are absent from the explicit split: {sorted(unassigned)}"
        )
    return (
        [by_id[trajectory_id] for trajectory_id in train_trajectory_ids],
        [by_id[trajectory_id] for trajectory_id in validation_trajectory_ids],
    )


def _resolve_trajectory_split(
    trajectories: list[Trajectory], config: AppConfig
) -> tuple[list[Trajectory], list[Trajectory]]:
    train_ids = config.training.train_trajectory_ids
    validation_ids = config.training.validation_trajectory_ids
    if train_ids is not None and validation_ids is not None:
        return _split_trajectories_explicit(trajectories, train_ids, validation_ids)
    return _split_trajectories(
        trajectories, config.training.validation_fraction, config.experiment.seed
    )


def _make_loader(
    dataset: TrajectoryWindowDataset, config: AppConfig, shuffle: bool
) -> DataLoader[dict[str, torch.Tensor]]:
    if len(dataset) == 0:
        raise ValueError("No valid samples; check trajectory lengths and history/forecast windows")
    generator = torch.Generator().manual_seed(config.experiment.seed)
    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=shuffle,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.num_workers > 0,
        generator=generator,
    )


def command_train(args: argparse.Namespace) -> int:
    config = _load(args.config)
    seed_everything(config.experiment.seed, config.experiment.deterministic)
    device = select_device(config.training.device)
    trajectories = load_trajectories(
        config.paths.trajectory_dir,
        config.data.feature_dim,
        config.data.num_stages,
    )
    if not trajectories and config.data.synthetic_if_missing:
        trajectories = make_synthetic_trajectories(
            config.data.synthetic_trajectories,
            config.data.synthetic_length,
            config.data.feature_dim,
            config.experiment.seed,
        )
    if not trajectories:
        raise FileNotFoundError(
            f"No trajectory .npz files in {config.paths.trajectory_dir}. "
            "Run the prepare command or use configs/experiment/smoke.yaml."
        )

    if args.fit_all:
        if config.training.train_trajectory_ids is not None:
            raise ValueError("--fit-all cannot be combined with an explicit trajectory split")
        train = list(trajectories)
        validation = list(trajectories)
    else:
        train, validation = _resolve_trajectory_split(trajectories, config)
    if len(trajectories) <= 3:
        print(
            json.dumps(
                {
                    "warning": "small trajectory split; validation metrics are not reliable",
                    "fit_all": args.fit_all,
                    "trajectory_count": len(trajectories),
                    "train_trajectory_ids": [item.trajectory_id for item in train],
                    "validation_trajectory_ids": [item.trajectory_id for item in validation],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    normalizer = StandardNormalizer.fit(train)
    normalizer.save(config.paths.artifact_root / config.experiment.name / "normalizer.npz")
    normalizer.save(config.paths.checkpoint_root / config.experiment.name / "normalizer.npz")
    train = [normalizer.transform(item) for item in train]
    validation = [normalizer.transform(item) for item in validation]
    train_dataset = TrajectoryWindowDataset(
        train,
        history_windows=config.data.history_windows,
        forecast_windows=config.data.forecast_windows,
    )
    validation_dataset = TrajectoryWindowDataset(
        validation,
        history_windows=config.data.history_windows,
        forecast_windows=config.data.forecast_windows,
    )
    stage_class_weights = None
    stage_class_counts = None
    if config.training.stage_class_weighting == "inverse_frequency":
        stage_class_weights, stage_class_counts = inverse_frequency_stage_weights(
            train_dataset, config.data.num_stages
        )
    if config.model.architecture == "latent_jepa":
        model: WorldModel = TemporalJEPAWorldModel(
            feature_dim=config.data.feature_dim,
            num_stages=config.data.num_stages,
            projection_dim=config.model.projection_dim,
            latent_dim=config.model.latent_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            min_log_variance=config.model.min_log_variance,
            max_log_variance=config.model.max_log_variance,
        )
    elif config.model.architecture == "temporal_transformer":
        model = TemporalTransformerWorldModel(
            feature_dim=config.data.feature_dim,
            num_stages=config.data.num_stages,
            projection_dim=config.model.projection_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            min_log_variance=config.model.min_log_variance,
            max_log_variance=config.model.max_log_variance,
        )
    elif config.model.architecture == "temporal_cnn":
        model = TemporalConvolutionalWorldModel(
            feature_dim=config.data.feature_dim,
            num_stages=config.data.num_stages,
            projection_dim=config.model.projection_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            min_log_variance=config.model.min_log_variance,
            max_log_variance=config.model.max_log_variance,
        )
    else:
        model = WorldModel(
            feature_dim=config.data.feature_dim,
            num_stages=config.data.num_stages,
            projection_dim=config.model.projection_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            min_log_variance=config.model.min_log_variance,
            max_log_variance=config.model.max_log_variance,
        )
    if config.training.compile and device.accelerator != "mps":
        model = cast(WorldModel, torch.compile(model))
    print(
        json.dumps(
            {
                "event": "training_start",
                "device": device.accelerator,
                "trajectories": len(trajectories),
                "train_samples": len(train_dataset),
                "validation_samples": len(validation_dataset),
                "train_trajectory_ids": [item.trajectory_id for item in train],
                "validation_trajectory_ids": [item.trajectory_id for item in validation],
                "small_trajectory_split": len(trajectories) <= 3,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "architecture": config.model.architecture,
                "stage_class_weighting": config.training.stage_class_weighting,
                "stage_class_counts": (
                    stage_class_counts.tolist() if stage_class_counts is not None else None
                ),
                "stage_class_weights": (
                    stage_class_weights.tolist() if stage_class_weights is not None else None
                ),
            },
            sort_keys=True,
        )
    )
    metrics = Trainer(
        model,
        config,
        device,
        stage_class_weights=stage_class_weights,
        stage_class_counts=stage_class_counts,
    ).fit(
        _make_loader(train_dataset, config, True),
        _make_loader(validation_dataset, config, False),
    )
    print(json.dumps({"event": "training_complete", "validation": metrics}, sort_keys=True))
    return 0


def command_prepare(args: argparse.Namespace) -> int:
    config = _load(args.config)
    if config.data.feature_dim != len(STATE_FEATURES):
        raise ValueError(
            f"Configured feature_dim={config.data.feature_dim}, but canonical schema has "
            f"{len(STATE_FEATURES)} features"
        )
    frame = read_flow_csv(Path(args.input), args.dataset)
    output = (
        Path(args.output)
        if args.output
        else (config.paths.trajectory_dir / f"{args.dataset}__{args.scenario}.npz")
    )
    build_trajectory(
        frame,
        trajectory_id=f"{args.dataset}:{args.scenario}",
        window_seconds=config.data.window_seconds,
        output_path=output,
    )
    print(json.dumps({"output": str(output), "flows": len(frame)}, sort_keys=True))
    return 0


def command_prepare_all(args: argparse.Namespace) -> int:
    config = _load(args.config)
    catalog = load_catalog(Path(args.catalog))
    prepared = []
    for spec in catalog.datasets:
        if not spec.enabled:
            continue
        if spec.dataset_id not in {"dapt2020", "cic_ids2018", "ctu13", "unsw_nb15"}:
            raise ValueError(f"Enabled dataset has no flow adapter: {spec.dataset_id}")
        matches = discover_files(spec, _repo_root())
        if not matches:
            print(
                json.dumps(
                    {
                        "warning": "enabled dataset matched no input files",
                        "dataset_id": spec.dataset_id,
                        "input_globs": spec.input_globs,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            continue
        for path in matches:
            scenario = spec.scenario_id(path)
            frame = read_flow_csv(path, spec.dataset_id)
            output = config.paths.trajectory_dir / f"{spec.dataset_id}__{scenario}.npz"
            build_trajectory(
                frame,
                trajectory_id=f"{spec.dataset_id}:{scenario}",
                window_seconds=config.data.window_seconds,
                output_path=output,
            )
            prepared.append(str(output))
    if not prepared:
        raise FileNotFoundError("No enabled dataset artifacts matched the catalog")
    print(json.dumps({"prepared": prepared, "count": len(prepared)}, indent=2))
    return 0


def command_prepare_pcap(args: argparse.Namespace) -> int:
    config = _load(args.config)
    frame = extract_pcap(
        Path(args.input),
        tcp_inactive_timeout=args.tcp_timeout,
        udp_inactive_timeout=args.udp_timeout,
        active_timeout=args.active_timeout,
        packet_limit=args.packet_limit,
    )
    output = Path(args.output) if args.output else (
        config.paths.trajectory_dir / f"{args.dataset}__{args.scenario}__pcap.npz"
    )
    build_trajectory(
        frame,
        trajectory_id=f"{args.dataset}:{args.scenario}:pcap",
        window_seconds=config.data.window_seconds,
        output_path=output,
    )
    print(json.dumps({"output": str(output), "reconstructed_flows": len(frame)}, sort_keys=True))
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    config = _load(args.config)
    selected = select_device(config.training.device)
    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "configured_device": config.training.device,
        "selected_device": selected.accelerator,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "mps_built": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_built()),
        "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        "trajectory_dir": str(config.paths.trajectory_dir),
        "trajectory_files": len(list(config.paths.trajectory_dir.glob("*.npz"))),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def command_explain(args: argparse.Namespace) -> int:
    config = _load(args.config)
    model, _ = load_model_from_checkpoint(args.checkpoint)
    normalizer_candidates = [
        config.paths.artifact_root / config.experiment.name / "normalizer.npz",
        Path(args.checkpoint).resolve().parent / "normalizer.npz",
    ]
    normalizer_path = next((path for path in normalizer_candidates if path.exists()), None)
    if normalizer_path is None:
        raise FileNotFoundError(
            "No training normalizer found. Expected artifacts/<experiment>/normalizer.npz "
            "or beside the checkpoint."
        )
    normalizer = StandardNormalizer.load(normalizer_path)
    if args.trajectory is None:
        raise ValueError("--trajectory is required for explain")
    trajectory_path = Path(args.trajectory)
    with np.load(trajectory_path, allow_pickle=False) as payload:
        states = np.asarray(payload["states"], dtype=np.float32)
        availability = np.asarray(
            payload.get("availability", np.ones_like(states)), dtype=np.float32
        )
    if args.window_index < 0 or args.window_index >= len(states):
        raise ValueError(
            f"window_index {args.window_index} is outside trajectory length {len(states)}"
        )

    history_start = max(0, args.window_index - config.data.history_windows + 1)
    history_real = states[history_start : args.window_index + 1]
    availability_real = availability[history_start : args.window_index + 1]
    pad = config.data.history_windows - len(history_real)
    if pad > 0:
        history_pad = np.zeros((pad, states.shape[1]), dtype=np.float32)
        availability_pad = np.zeros((pad, states.shape[1]), dtype=np.float32)
        history_real = np.concatenate([history_pad, history_real], axis=0)
        availability_real = np.concatenate([availability_pad, availability_real], axis=0)

    raw_history = history_real.copy()
    normalized_history = (history_real - normalizer.mean) / normalizer.scale
    normalized_history *= availability_real
    history = torch.from_numpy(normalized_history).unsqueeze(0)
    availability = torch.from_numpy(availability_real).unsqueeze(0)
    rollout = model.rollout(history, availability, config.data.forecast_windows)
    results: list[dict[str, object]] = []
    for step in range(config.data.forecast_windows):
        stage_index = int(torch.argmax(rollout.stage_logits[0, step]).item())
        malicious_score = float(torch.sigmoid(rollout.malicious_logits[0, step]).item())
        compromise_score = float(torch.sigmoid(rollout.compromise_logits[0, step]).item())
        attribution = attribute_prediction(
            model,
            history[0],
            availability[0],
            target_head="malicious_risk",
            forecast_step=step,
            n_steps=50,
        )
        ranked = sorted(
            (
                {
                    "feature": feature_name,
                    "score": float(score),
                    "raw_value": float(raw_history[-1, idx]),
                }
                for idx, (feature_name, score) in enumerate(
                    zip(STATE_FEATURES, attribution, strict=True)
                )
            ),
            key=lambda entry: abs(entry["score"]),
            reverse=True,
        )[: args.top_k]
        results.append(
            {
                "forecast_step": step,
                "predicted_stage": stage_index,
                "malicious_risk": malicious_score,
                "compromise_risk": compromise_score,
                "attribution_space": "normalized",
                "top_features": ranked,
            }
        )
    output_path = config.paths.run_root / config.experiment.name / (
        f"explain_{trajectory_path.stem}_{args.window_index}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


def command_benchmark(args: argparse.Namespace) -> int:
    config = _load(args.config)
    trajectory_dir = Path(args.dataset_dir) if args.dataset_dir else config.paths.trajectory_dir
    trajectories = load_trajectories(
        trajectory_dir,
        config.data.feature_dim,
        config.data.num_stages,
    )
    if not trajectories and config.data.synthetic_if_missing:
        trajectories = make_synthetic_trajectories(
            config.data.synthetic_trajectories,
            config.data.synthetic_length,
            config.data.feature_dim,
            config.experiment.seed,
        )
    if not trajectories:
        raise FileNotFoundError(
            f"No trajectory .npz files in {trajectory_dir}. Configure a dataset dir or run prepare."
        )

    train, validation = _resolve_trajectory_split(trajectories, config)
    evaluation = validation
    if args.risk_trajectory:
        risk_path = Path(args.risk_trajectory).resolve()
        candidates = load_trajectories(
            trajectory_dir,
            config.data.feature_dim,
            config.data.num_stages,
        )
        selected = [
            item
            for item in candidates
            if item.trajectory_id.replace(":", "__") == risk_path.stem
        ]
        if not selected:
            raise FileNotFoundError(f"Could not find trajectory {risk_path}")
        evaluation = selected
        selected_ids = {item.trajectory_id for item in selected}
        trajectories_for_fit = [
            item for item in trajectories if item.trajectory_id not in selected_ids
        ]
        if len(trajectories_for_fit) < 2:
            raise ValueError(
                "Explicit risk trajectory requires at least two separate fit trajectories"
            )
        train = trajectories_for_fit
    normalizer = StandardNormalizer.fit(train)
    train_samples = make_baseline_samples(
        [normalizer.transform(item) for item in train],
        history_windows=config.data.history_windows,
        forecast_windows=config.data.forecast_windows,
    )
    evaluation_samples = make_baseline_samples(
        [normalizer.transform(item) for item in evaluation],
        history_windows=config.data.history_windows,
        forecast_windows=config.data.forecast_windows,
    )
    model = train_logistic_baseline(train_samples, target=args.target)
    baseline_labels = (
        evaluation_samples.malicious
        if args.target == "malicious"
        else evaluation_samples.compromise
    )
    baseline_predictions = model.predict(evaluation_samples.features)
    baseline_metrics = compute_metrics(baseline_labels, baseline_predictions)

    world_model, _ = load_model_from_checkpoint(args.checkpoint)
    world_model.eval()
    world_logits: list[float] = []
    world_compromise_logits: list[float] = []
    for history, availability in zip(
        evaluation_samples.histories,
        evaluation_samples.history_availabilities,
        strict=True,
    ):
        with torch.no_grad():
            rollout = world_model.rollout(
                torch.from_numpy(history).unsqueeze(0),
                torch.from_numpy(availability).unsqueeze(0),
                1,
            )
        logit = (
            rollout.malicious_logits[0, 0]
            if args.target == "malicious"
            else rollout.compromise_logits[0, 0]
        )
        world_logits.append(float(logit.item()))
        world_compromise_logits.append(float(rollout.compromise_logits[0, 0].item()))
    world_predictions = (torch.sigmoid(torch.tensor(world_logits)) >= 0.5).numpy().astype(np.int64)
    world_metrics = compute_metrics(baseline_labels, world_predictions)
    compromise_labels = evaluation_samples.compromise
    if np.unique(train_samples.compromise).size < 2:
        compromise_baseline = {
            "status": "unavailable_one_class_fit_labels",
            "support": len(compromise_labels),
        }
    else:
        compromise_model = train_logistic_baseline(train_samples, target="compromise")
        compromise_predictions = compromise_model.predict(evaluation_samples.features)
        compromise_baseline = compute_metrics(compromise_labels, compromise_predictions)
    compromise_predictions = (
        torch.sigmoid(torch.tensor(world_compromise_logits)) >= 0.5
    ).numpy().astype(np.int64)
    compromise_world = compute_metrics(compromise_labels, compromise_predictions)
    per_feature_error = np.zeros(config.data.feature_dim, dtype=np.float64)
    per_feature_count = np.zeros(config.data.feature_dim, dtype=np.float64)
    for history, availability, target, target_availability in zip(
        evaluation_samples.histories,
        evaluation_samples.history_availabilities,
        evaluation_samples.future_states,
        evaluation_samples.future_availabilities,
        strict=True,
    ):
        with torch.no_grad():
            prediction = world_model.rollout(
                torch.from_numpy(history).unsqueeze(0),
                torch.from_numpy(availability).unsqueeze(0),
                1,
            ).state_mean[0, 0].numpy()
        squared = (prediction - target) ** 2
        per_feature_error += squared * target_availability
        per_feature_count += target_availability
    per_feature_mse = per_feature_error / np.maximum(per_feature_count, 1.0)
    feature_names = STATE_FEATURES[: config.data.feature_dim]
    shared_indices = [
        index
        for index in range(config.data.feature_dim)
        if any(item.availability[:, index].any() for item in train)
    ]
    cic_unobserved_indices = [
        index for index in range(config.data.feature_dim) if index not in shared_indices
    ]

    def feature_group(indices: list[int]) -> dict[str, object]:
        observations = float(per_feature_count[indices].sum()) if indices else 0.0
        squared_error = float(per_feature_error[indices].sum()) if indices else 0.0
        return {
            "feature_count": len(indices),
            "observations": int(observations),
            "aggregate_mse": squared_error / max(1.0, observations),
            "aggregate_rmse": float(np.sqrt(squared_error / max(1.0, observations))),
            "per_feature_mse": {
                feature_names[index]: float(per_feature_mse[index]) for index in indices
            },
        }

    state_shift = {
        "scope": "cross_dataset_generalization",
        "trajectory_ids": [item.trajectory_id for item in evaluation],
        "valid_risk_rows": len(evaluation_samples.features),
        "state_rmse": float(
            np.sqrt(per_feature_error.sum() / max(1.0, per_feature_count.sum()))
        ),
        "feature_groups": {
            "shared": feature_group(shared_indices),
            "cic_unobserved": feature_group(cic_unobserved_indices),
        },
    }
    output_dir = config.paths.run_root / config.experiment.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "benchmark.json"
    payload = {
        "event": "benchmark_complete",
        "target": args.target,
        "train_trajectories": len(train),
        "validation_trajectories": len(validation),
        "evaluation_scope": "risk_trajectory" if args.risk_trajectory else "validation_split",
        "evaluation_trajectory_ids": [item.trajectory_id for item in evaluation],
        "baseline": baseline_metrics,
        "world_model": world_metrics,
        "risk_metrics": {
            "malicious_risk": {
                "baseline": baseline_metrics,
                "world_model": world_metrics,
            },
            "compromise_risk": {
                "baseline": compromise_baseline,
                "world_model": compromise_world,
            },
        },
        "validation_split_info": {
            "train_trajectories": len(train),
            "validation_trajectories": len(validation),
            "valid_prediction_rows": len(evaluation_samples.features),
        },
        "excluded_invalid_label_count": sum(
            len(
                TrajectoryWindowDataset(
                    [item],
                    history_windows=config.data.history_windows,
                    forecast_windows=config.data.forecast_windows,
                )
            )
            * config.data.forecast_windows
                for item in evaluation
            ) - len(evaluation_samples.features),
        "state_shift_report": state_shift if args.state_shift_report else None,
        "output": str(output_path),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("model,precision,recall,f1,fpr")
    for name, result in (("baseline", baseline_metrics), ("world_model", world_metrics)):
        values = [f"{result[key]:.4f}" for key in ("precision", "recall", "f1", "fpr")]
        print(",".join([name, *values]))
    print(json.dumps(payload, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cyberworld")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="Train a world model")
    train.add_argument("--config", default="configs/experiment/base.yaml")
    train.add_argument(
        "--fit-all",
        action="store_true",
        help="Fit and validate on every discovered trajectory without a holdout split",
    )
    train.set_defaults(handler=command_train)

    prepare = subparsers.add_parser("prepare", help="Convert one flow CSV into a trajectory")
    prepare.add_argument("--config", default="configs/experiment/base.yaml")
    prepare.add_argument(
        "--dataset", required=True, choices=("dapt2020", "cic_ids2018", "ctu13", "unsw_nb15")
    )
    prepare.add_argument("--scenario", required=True)
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output")
    prepare.set_defaults(handler=command_prepare)

    prepare_all = subparsers.add_parser(
        "prepare-all", help="Prepare every enabled artifact in a dataset catalog"
    )
    prepare_all.add_argument("--config", default="configs/experiment/base.yaml")
    prepare_all.add_argument("--catalog", default="configs/datasets/catalog.yaml")
    prepare_all.set_defaults(handler=command_prepare_all)

    prepare_pcap = subparsers.add_parser(
        "prepare-pcap", help="Extract packet-derived features into a trajectory"
    )
    prepare_pcap.add_argument("--config", default="configs/experiment/base.yaml")
    prepare_pcap.add_argument("--dataset", required=True)
    prepare_pcap.add_argument("--scenario", required=True)
    prepare_pcap.add_argument("--input", required=True)
    prepare_pcap.add_argument("--output")
    prepare_pcap.add_argument("--packet-limit", type=int)
    prepare_pcap.add_argument("--tcp-timeout", type=float, default=120.0)
    prepare_pcap.add_argument("--udp-timeout", type=float, default=60.0)
    prepare_pcap.add_argument("--active-timeout", type=float, default=300.0)
    prepare_pcap.set_defaults(handler=command_prepare_pcap)

    doctor = subparsers.add_parser("doctor", help="Report runtime and accelerator support")
    doctor.add_argument("--config", default="configs/experiment/base.yaml")
    doctor.set_defaults(handler=command_doctor)

    explain = subparsers.add_parser("explain", help="Run attribution on a trajectory window")
    explain.add_argument("--config", default="configs/experiment/base.yaml")
    explain.add_argument("--checkpoint", required=True)
    explain.add_argument("--trajectory", required=True)
    explain.add_argument("--window-index", type=int, required=True)
    explain.add_argument("--top-k", type=int, default=5)
    explain.set_defaults(handler=command_explain)

    benchmark = subparsers.add_parser(
        "benchmark",
        help="Evaluate a logistic-regression baseline on flattened trajectory states",
    )
    benchmark.add_argument("--config", default="configs/experiment/base.yaml")
    benchmark.add_argument("--checkpoint", required=True)
    benchmark.add_argument("--dataset-dir", default=None)
    benchmark.add_argument("--target", choices=("malicious", "compromise"), default="malicious")
    benchmark.add_argument("--risk-trajectory", default=None)
    benchmark.add_argument("--state-shift-report", action="store_true")
    benchmark.set_defaults(handler=command_benchmark)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
