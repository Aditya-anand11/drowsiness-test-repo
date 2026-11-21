"""Standalone training launcher for the driver-drowsiness models.

Why this script exists
----------------------
``torch`` / ``torchvision`` are heavy (hundreds of MB of wheels, multi-GB of RAM
for a real run).  TensorFlow is **not** used anywhere in this codebase -- every
model (``models_cnn.py``, ``models_temporal.py``, ``models_vae.py``) is written
in PyTorch -- so there is nothing to train with TensorFlow here and installing
it would only waste disk and RAM.  Both packages are therefore *not* installed
by this script and cannot be installed inside the restricted sandbox
(512 MB RAM, 1 CPU, no network, no GPU).

Run this script on a dedicated machine instead::

    python -m pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu
    python run_training.py --dry-run              # wiring check, no torch needed
    python run_training.py --model cnn            # train CNN2D for a few epochs
    python run_training.py --model temporal       # train the LSTM
    python run_training.py --model vae            # train the VAE

Torch is imported lazily so ``--dry-run`` works with no heavy dependency.
"""

from __future__ import annotations

import argparse
import sys


INSTALL_HINT = (
    "PyTorch is not installed, so training cannot run here.\n"
    "Install the CPU build on a dedicated machine:\n"
    "    python -m pip install torch torchvision "
    "--index-url https://download.pytorch.org/whl/cpu\n"
    "(TensorFlow is not used by this repo; every model is PyTorch.)"
)


def _import_torch():
    """Return the torch module, or None if it is not installed."""
    try:
        import torch  # noqa: WPS433 - delayed import on purpose
        return torch
    except Exception:
        return None


def _check_data_dirs():
    """Report which image folders the dataloader expects but cannot find."""
    import os

    expected = ["train", "test", "no yawn", "yawn"]
    missing = [d for d in expected if not os.path.isdir(d)]
    return expected, missing


def dry_run() -> int:
    """Validate wiring without importing torch or touching any data."""
    print("run_training.py -- dry-run")
    print(f"  python: {sys.version.split()[0]}")

    torch = _import_torch()
    if torch is None:
        print("  torch: NOT installed (expected in the sandbox)")
        print(INSTALL_HINT)
    else:
        print(f"  torch: {getattr(torch, '__version__', 'unknown')}")
        print(f"  cuda available: {torch.cuda.is_available()}")

    expected, missing = _check_data_dirs()
    print(f"  expected data dirs: {', '.join(expected)}")
    if missing:
        print(f"  missing data dirs: {', '.join(missing)}")
    else:
        print("  all data dirs present")

    # Verify the training entry points are importable *by name* only; importing
    # train.py itself would import torch at module level.
    import importlib.util

    for module in ("train", "dataloader", "models_cnn", "models_temporal", "models_vae"):
        spec = importlib.util.find_spec(module)
        print(f"  module {module}: {'found' if spec else 'MISSING'}")

    print("dry-run complete.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("cnn", "temporal", "vae"),
        default="cnn",
        help="which training routine in train.py to run",
    )
    parser.add_argument("--epochs", type=int, default=3, help="number of epochs")
    parser.add_argument(
        "--dataset",
        default="both",
        help="dataset_type passed to get_loaders (e.g. both / eye / yawn)",
    )
    parser.add_argument("--optimizer", default="adam", help="cnn optimizer: sgd/momentum/nesterov/adam")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="check wiring only; does not import torch or read data",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        return dry_run()

    if _import_torch() is None:
        print(INSTALL_HINT)
        return 1

    expected, missing = _check_data_dirs()
    if missing:
        print("Missing required data directories: " + ", ".join(missing))
        print("Create them (or point the loader at your dataset) before training.")
        return 1

    # Imported only after torch is known to be present.
    from train import train_ablation_cnn, train_temporal, train_vae

    if args.model == "cnn":
        model = train_ablation_cnn(
            dataset_type=args.dataset,
            opt_name=args.optimizer,
            epochs=args.epochs,
        )
    elif args.model == "temporal":
        model = train_temporal(epochs=args.epochs)
    else:
        model = train_vae(dataset_type=args.dataset, epochs=args.epochs)

    print(f"Training finished: {type(model).__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
