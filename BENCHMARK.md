# Running the benchmark

`benchmark.py` measures dataloader throughput, training throughput
(forward + backward samples/second) and inference FPS/latency for the models in
this repo.

## Requirements

Install the dependencies (these are the same libraries the source files import):

```
pip install -r requirements.txt
```

`benchmark.py` also needs the image folders used by `dataloader.get_loaders`:
`train/`, `test/`, `no yawn/`, `yawn/` in the working directory.

> Note: PyTorch cannot be imported inside the restricted CI/sandbox container
> (memory-capped, no GPU). Run the benchmark on a dedicated machine or GPU box.

## Usage

```
python benchmark.py
python benchmark.py --dataset-type eyes --batch-size 64 --epochs 2
python benchmark.py --skip-training --inference-runs 100
python benchmark.py --output results/gpu_run.json
```

Key options:

- `--dataset-type {both,eyes,yawn}` - which loader to build.
- `--batch-size` - batch size for the loaders and temporal training.
- `--epochs` - training epochs timed per model.
- `--train-batches` - cap on batches timed per epoch (0 = all).
- `--inference-runs` - number of timed forward passes for FPS/latency.
- `--skip-training` - only dataloader + inference measurements.
- `--output` - path of the JSON report (default `benchmark_report.json`).

The JSON report records the device, PyTorch version, CUDA availability, the
configuration used, dataset sizes, dataloader samples/second, per-epoch training
samples/second, and inference FPS/latency, so runs across machines can be
compared.
