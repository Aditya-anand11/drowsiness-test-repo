"""Standalone benchmark harness for the driver-drowsiness models in this repo.

It measures three things that matter when training the models in ``train.py``:

1. Data pipeline throughput  - how many (image, label) samples per second the
   loaders from ``dataloader.py`` can deliver (``get_loaders``).
2. Training throughput       - how many samples per second a forward + backward
   pass of the CNN models processes (``CNN2D``, ``ResNet``, ``GoogLeNet``),
   plus the same for ``TemporalLSTM`` on the synthetic sequence loader.
3. Inference throughput      - FPS and latency per sample at batch size 1 for
   every model (uses ``evaluate.benchmark_fps`` for the CNNs).

The script writes a JSON report (default ``benchmark_report.json``) so results
from different machines / GPUs can be diffed.

IMPORTANT: this requires PyTorch, torchvision, Pillow, NumPy, SciKit-Learn and
the image folders used by ``dataloader.get_loaders`` (``train/``, ``test/``,
``no yawn/``, ``yawn/``). See BENCHMARK.md for how to run it on a dedicated
machine.
"""

import argparse
import json
import os
import time


def import_stack():
    try:
        import torch  # noqa: F401
    except ImportError:
        raise SystemExit(
            "PyTorch is required for the benchmark and is not importable in this "
            "environment. Run benchmark.py on a machine with the dependencies "
            "from requirements.txt installed (see BENCHMARK.md)."
        )
    import torch
    from dataloader import get_loaders, get_sequence_loader
    from models_cnn import CNN2D, ResNet, GoogLeNet
    from models_temporal import TemporalLSTM
    from evaluate import benchmark_fps

    return {
        "torch": torch,
        "get_loaders": get_loaders,
        "get_sequence_loader": get_sequence_loader,
        "CNN2D": CNN2D,
        "ResNet": ResNet,
        "GoogLeNet": GoogLeNet,
        "TemporalLSTM": TemporalLSTM,
        "benchmark_fps": benchmark_fps,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark the drowsiness models.")
    parser.add_argument("--dataset-type", default="both",
                        choices=["both", "eyes", "yawn"],
                        help="Which loader to build (maps to get_loaders(dataset_type=...)).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1,
                        help="Epochs of training timing per model.")
    parser.add_argument("--train-batches", type=int, default=20,
                        help="Max training batches timed per epoch (0 = all).")
    parser.add_argument("--inference-runs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--output", default="benchmark_report.json")
    parser.add_argument("--skip-training", action="store_true",
                        help="Only measure inference throughput / dataloader speed.")
    return parser.parse_args()


def dataloader_throughput(train_loader, max_batches):
    """Iterate the loader once and report samples/second."""
    start = time.time()
    seen = 0
    for i, (images, _) in enumerate(train_loader):
        seen += images.size(0)
        if max_batches and i + 1 >= max_batches:
            break
    elapsed = time.time() - start
    return {"samples": seen, "seconds": elapsed,
            "samples_per_second": seen / elapsed if elapsed > 0 else 0.0}


def train_throughput(stack, model, loader, device, epochs, max_batches, lr):
    """Time forward+backward passes and report samples/second."""
    torch = stack["torch"]
    import torch.nn as nn

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    history = []

    for epoch in range(epochs):
        model.train()
        start = time.time()
        seen = 0
        for i, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            labels = labels.float()
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            seen += inputs.size(0)
            if max_batches and i + 1 >= max_batches:
                break
        elapsed = time.time() - start
        history.append({
            "epoch": epoch + 1,
            "samples": seen,
            "seconds": elapsed,
            "samples_per_second": seen / elapsed if elapsed > 0 else 0.0,
        })
    return history


def sequence_throughput(stack, device, batch_size, epochs, lr):
    """Train the temporal model on the synthetic sequence loader."""
    torch = stack["torch"]
    import torch.nn as nn

    loader = stack["get_sequence_loader"](batch_size=batch_size)
    model = stack["TemporalLSTM"]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    history = []

    for epoch in range(epochs):
        model.train()
        start = time.time()
        seen = 0
        for seqs, labels in loader:
            seqs = seqs.to(device)
            labels = labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            seen += seqs.size(0)
        elapsed = time.time() - start
        history.append({
            "epoch": epoch + 1,
            "samples": seen,
            "seconds": elapsed,
            "samples_per_second": seen / elapsed if elapsed > 0 else 0.0,
        })
    return history


def main():
    args = parse_args()
    stack = import_stack()
    torch = stack["torch"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_size = (args.img_size, args.img_size)
    report = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "config": vars(args),
        "dataloader": {},
        "training": {},
        "inference": {},
    }

    print(f"[benchmark] device={device} torch={torch.__version__}")

    train_loader, test_loader = stack["get_loaders"](
        dataset_type=args.dataset_type, img_size=img_size,
        batch_size=args.batch_size, augment=True,
    )
    report["dataset"] = {
        "train_samples": len(train_loader.dataset),
        "test_samples": len(test_loader.dataset),
    }

    print("[benchmark] measuring dataloader throughput ...")
    report["dataloader"]["train"] = dataloader_throughput(
        train_loader, args.train_batches)
    print("    ", report["dataloader"]["train"])

    models = {
        "CNN2D": stack["CNN2D"](),
        "ResNet": stack["ResNet"](),
        "GoogLeNet": stack["GoogLeNet"](),
    }

    if not args.skip_training:
        for name, model in models.items():
            print(f"[benchmark] training {name} for {args.epochs} epoch(s) ...")
            report["training"][name] = train_throughput(
                stack, model, train_loader, device, args.epochs,
                args.train_batches, args.lr)

        print("[benchmark] training TemporalLSTM on synthetic sequences ...")
        report["training"]["TemporalLSTM"] = sequence_throughput(
            stack, device, args.batch_size, args.epochs, args.lr)

    print("[benchmark] measuring inference FPS/latency ...")
    input_shape = (1, 3, args.img_size, args.img_size)
    for name, model in models.items():
        model = model.to(device)
        fps_stats = stack["benchmark_fps"](
            model, input_shape=input_shape,
            num_runs=args.inference_runs, device=str(device))
        report["inference"][name] = fps_stats
        print(f"    {name}: {fps_stats}")

    with open(args.output, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[benchmark] report written to {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
