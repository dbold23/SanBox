"""End-to-end CLI smoke on a small synthetic corpus, zero optional deps."""

from __future__ import annotations

import json
import os
import subprocess
import sys

PROTO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_cli_end_to_end(tmp_path):
    root = str(tmp_path / "corpus")
    out = str(tmp_path / "results")
    cmd = [
        sys.executable,
        os.path.join(PROTO_DIR, "run_ablation.py"),
        "--data", "synthetic",
        "--root", root,
        "--backbone", "hist",
        "--arms", "head,body,headless,cross_orientation",
        "--out", out,
        "--seed", "3",
        "--n-individuals", "25",
    ]
    proc = subprocess.run(cmd, cwd=PROTO_DIR, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr

    with open(os.path.join(out, "results.json")) as f:
        results = json.load(f)
    assert set(results["arms"]) == {"head", "body", "headless", "cross_orientation"}
    for arm, metrics in results["arms"].items():
        assert metrics["n_gallery"] > 0
        assert len(metrics["rejection_curve"]) > 0
    assert results["verdict"]["verdict"] is not None
    assert "board-mounted" in results["caveat"]

    with open(os.path.join(out, "report.md")) as f:
        report = f.read()
    assert results["verdict"]["verdict"] in report
    assert "board-mounted" in report
    assert "VERDICT:" in proc.stdout
    assert "board-mounted" in proc.stdout
