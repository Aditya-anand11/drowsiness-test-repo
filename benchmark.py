"""Standalone benchmark + training-smoke harness for the drowsiness models.

This script is intentionally self-contained so it can be copied to (and run on)
a dedicated machine with PyTorch installed.  It measures inference throughput /
latency for the CNN models defined in ``models_cnn.py`` and, optionally, times a
few training steps of ``CNN2D`` so you can tell whether a model is trainable and
how fast before committing to a full run.

Why a standalone script instead of touching CI?
-----------------------------------------------
``torch`` / ``torchvision`` are heavy (hundreds of MB, multi-GB of RAM for a
real training run).  They cannot be installed in the restricted sandbox, so the
benchmark lives here and is executed on a real machine::

    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    python benchmark.py                     # full benchmark (needs torch)
    python benchmark.py --train-steps 20    # also time 20 training steps
    python benchmark.py --json out.json     # machine-readable report
    python benchmark.py --dry-run           # no torch needed: sanity-check wiring

The script never imports torch at module import time, so ``--dry-run`` works in
the sandbox and on any machine without the heavy dependency.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from typing import Any, Dict, List


# --------------------------------------------------------------------------- #
# Torch availability
# --------------------------------------------------------------------------- #
def _import_torch():
    """Return the torch module, or None if it is not installed."""
    try:
        import torch  # noqa: WPS433 (delayed import on purpose)
        return torch
    except Exception:  # pragma: no cover - depends on environment
        return None


INSTALL_HINT = (
    "PyTorch is not installed, so the real benchmark cannot run here.\n"
    "On a dedicated machine install the CPU build with:\n"
    "    python -m pip install torch torchvision "
    "--index-url https://download.pytorch.org/whl/cpu\n"
    "then run:  python benchmark.py"
)


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
# Only models that are actually defined in models_cnn.py are referenced here.
# ``kwargs`` mirror how train.py / evaluate.py construct each model.
MODEL_SPECS: List[Dict[str, Any]] = [
    {"name": "CNN2D", "kwargs": {"use_bn": True, "dropout": 0.5, "init_type": "he"}},
    {"name": "ResNet", "kwargs": {}},
    {"name": "GoogLeNet", "kwargs": {}},
]


def build_model(torch, name: str, kwargs: Dict[str, Any]):
    """Instantiate a model from models_cnn by name."""
    import models_cnn

    cls = getattr(models_cnn, name, None)
    if cls is None:
        return None
    return cls(**kwargs)


# --------------------------------------------------------------------------- #
# Benchmarks
# --------------------------------------------------------------------------- #
def benchmark_inference(torch, model, input_shape, num_runs=50, warmup=5, device="cpu"):
    """Time forward passes.  Mirrors evaluate.benchmark_fps but batch-aware."""
    model.eval()
    dummy = torch.randn(*input_shape).to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        start = time.perf_counter()
        for _ in range(num_runs):
            model(dummy)
        elapsed = time.perf_counter() - start
    batch = input_shape[0]
    images = num_runs * batch
    return {
        "batch_size": batch,
        "runs": num_runs,
        "fps": images / elapsed,
        "latency_ms_per_batch": (elapsed / num_runs) * 1000.0,
        "latency_ms_per_image": (elapsed / images) * 1000.0,
    }


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_training(torch, steps=10, batch_size=8, device="cpu"):
    """Time a handful of CNN2D training steps (optimizer + backward included)."""
    import torch.nn as nn

    from models_cnn import CNN2D

    model = CNN2D(use_bn=True, dropout=0.5, init_type="he").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    images = torch.randn(batch_size, 3, 64, 64, device=device)
    # train.py feeds labels as float of shape (B, 1) via unsqueeze(1).
    labels = torch.randint(0, 2, (batch_size, 1), device=device).float()

    model.train()
    start = time.perf_counter()
    last_loss = float("nan")
    for _ in range(steps):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        last_loss = loss.item()
    elapsed = time.perf_counter() - start
    return {
        "steps": steps,
        "batch_size": batch_size,
        "seconds": elapsed,
        "steps_per_second": steps / elapsed,
        "last_loss": last_loss,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--batch-size", type=int, default=1, help="inference batch size (default 1)")
    p.add_argument("--img-size", type=int, default=64, help="square input size (default 64)")
    p.add_argument("--runs", type=int, default=50, help="timed forward passes (default 50)")
    p.add_argument("--warmup", type=int, default=5, help="warmup forward passes (default 5)")
    p.add_argument("--device", default="cpu", help="torch device, e.g. cpu or cuda")
    p.add_argument("--train-steps", type=int, default=0, help="also time N CNN2D training steps")
    p.add_argument("--train-batch-size", type=int, default=8)
    p.add_argument("--json", default=None, help="write the report as JSON to this path")
    p.add_argument("--dry-run", action="store_true", help="check wiring without running the benchmark")
    p.add_argument("--models", default=None, help="comma-separated subset of model names")
    return p.parse_args(argv)


def selected_specs(raw: str | None) -> List[Dict[str, Any]]:
    if not raw:
        return MODEL_SPECS
    wanted = {name.strip() for name in raw.split(",") if name.strip()}
    return [spec for spec in MODEL_SPECS if spec["name"] in wanted]


def run_dry(torch) -> Dict[str, Any]:
    """Validate wiring (module names, script structure) without heavy compute."""
    report: Dict[str, Any] = {
        "mode": "dry-run",
        "torch_available": torch is not None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "models": [spec["name"] for spec in MODEL_SPECS],
    }
    if torch is None:
        report["note"] = "torch missing; only the script wiring was checked"
        return report
    import models_cnn

    found = [spec["name"] for spec in MODEL_SPECS if hasattr(models_cnn, spec["name"])]
    report["models_found_in_models_cnn"] = found
    return report


def main(argv=None) -> int:
    args = parse_args(argv)
    torch = _import_torch()

    if args.dry_run:
        report = run_dry(torch)
        print(json.dumps(report, indent=2))
        return 0

    if torch is None:
        print(INSTALL_HINT, file=sys.stderr)
        return 2

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU", file=sys.stderr)
        device = torch.device("cpu")

    input_shape = (args.batch_size, 3, args.img_size, args.img_size)
    report: Dict[str, Any] = {
        "mode": "benchmark",
        "device": str(device),
        "input_shape": list(input_shape),
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "results": [],
    }

    for spec in selected_specs(args.models):
        try:
            model = build_model(torch, spec["name"], spec["kwargs"])
        except Exception as exc:  # pragma: no cover - depends on models_cnn
            report["results"].append({"model": spec["name"], "error": repr(exc)})
            continue
        if model is None:
            report["results"].append({"model": spec["name"], "error": "not found in models_cnn"})
            continue
        model = model.to(device)
        timing = benchmark_inference(
            torch, model, input_shape, num_runs=args.runs, warmup=args.warmup, device=device
        )
        entry = {"model": spec["name"], "parameters": count_parameters(model), **timing}
        report["results"].append(entry)
        print(
            "{model:<12} params={parameters:>10,}  fps={fps:8.1f}  "
            "latency={latency_ms_per_image:7.3f} ms/img".format(**entry)
        )

    if args.train_steps > 0:
        try:
            report["training"] = benchmark_training(
                torch, steps=args.train_steps, batch_size=args.train_batch_size, device=device
            )
            t = report["training"]
            print(
                "CNN2D train  {steps} steps in {seconds:.3f}s  "
                "({steps_per_second:.2f} steps/s, last loss {last_loss:.4f})".format(**t)
            )
        except Exception as exc:  # pragma: no cover - depends on models_cnn
            report["training"] = {"error": repr(exc)}
            print("training benchmark failed:", exc, file=sys.stderr)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print("report written to", args.json)

    failures = [r for r in report["results"] if "error" in r]
    return 1 if failures and len(failures) == len(report["results"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
