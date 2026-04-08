#!/usr/bin/env python3
"""CLI entry point for hybrid QCNN training."""

from __future__ import annotations

import argparse
from pathlib import Path

from quantum_cnn.config import QuantumCNNConfig
from quantum_cnn.run_experiment import run_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Train hybrid quantum-classical QCNN")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hybrid_qcnn.json",
        help="Path to JSON config file",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Optional override for experiment output directory name",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = QuantumCNNConfig.from_json(Path(args.config))
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    run_experiment(config)


if __name__ == "__main__":
    main()
