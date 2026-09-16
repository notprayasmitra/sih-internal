"""Command-line entry points for data preparation, diagnostics, and training."""

from __future__ import annotations

import argparse
import json
import platform
import random
from pathlib import Path
from typing import cast

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from cyberworld.adapters import read_flow_csv
from cyberworld.config import AppConfig, load_config
from cyberworld.data.catalog import discover_files, load_catalog
from cyberworld.data.normalization import StandardNormalizer
from cyberworld.data.synthetic import make_synthetic_trajectories
from cyberworld.data.trajectory import Trajectory, TrajectoryWindowDataset, load_trajectories
from cyberworld.models import TemporalJEPAWorldModel, WorldModel
from cyberworld.pcap import extract_pcap
from cyberworld.preprocessing import STATE_FEATURES, build_trajectory
from cyberworld.runtime import seed_everything, select_device
from cyberworld.training import Trainer


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

    train, validation = _split_trajectories(
        trajectories, config.training.validation_fraction, config.experiment.seed
    )
    normalizer = StandardNormalizer.fit(train)
    normalizer.save(config.paths.artifact_root / config.experiment.name / "normalizer.npz")
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
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "architecture": config.model.architecture,
            },
            sort_keys=True,
        )
    )
    metrics = Trainer(model, config, device).fit(
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
        for path in discover_files(spec, _repo_root()):
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cyberworld")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="Train a world model")
    train.add_argument("--config", default="configs/experiment/base.yaml")
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
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
