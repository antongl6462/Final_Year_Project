#!/usr/bin/env python3
"""Run baseline-vs-QCNN comparison using existing repository scripts.

This script does not modify the classical pipeline; it only orchestrates calls.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from quantum_cnn.config import QuantumCNNConfig
from quantum_cnn.run_experiment import run_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Compare YOLO baseline and hybrid QCNN")
    parser.add_argument("--data_root", type=str, required=True, help="Root containing classify_yolo/")
    parser.add_argument("--classical_model", type=str, required=True, help="Path to trained YOLO .pt")
    parser.add_argument("--qcnn_config", type=str, default="configs/hybrid_qcnn.json")
    parser.add_argument("--output", type=str, default="runs/quantum_cnn/comparison_summary.json")
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


def run_classical_eval(repo_root: Path, args) -> dict:
    output_dir = repo_root / "runs" / "classify" / "qcnn_comparison_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "src/eval_classify.py",
        "--model",
        str(Path(args.classical_model).resolve()),
        "--data_root",
        str(Path(args.data_root).resolve()),
        "--split",
        "test",
        "--output_dir",
        str(output_dir),
    ]
    if args.device:
        cmd.extend(["--device", args.device])

    subprocess.run(cmd, check=True, cwd=str(repo_root))

    with open(output_dir / "metrics_test.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    print("Running classical YOLO evaluation...")
    classical_metrics = run_classical_eval(repo_root, args)

    print("Running hybrid QCNN experiment...")
    qcnn_config = QuantumCNNConfig.from_json(Path(args.qcnn_config))
    qcnn_results = run_experiment(qcnn_config)

    summary = {
        "note": "Hybrid QCNN is a simulation proof-of-concept; no quantum advantage claim is made.",
        "classical": {
            "accuracy": classical_metrics.get("accuracy"),
            "roc_auc": classical_metrics.get("roc_auc"),
            "metrics_path": "runs/classify/qcnn_comparison_eval/metrics_test.json",
        },
        "hybrid_qcnn": {
            "accuracy": qcnn_results["metrics"].get("accuracy"),
            "roc_auc": qcnn_results["metrics"].get("roc_auc"),
            "artifacts_dir": qcnn_results["output_dir"],
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved comparison summary to: {output_path}")


if __name__ == "__main__":
    main()
