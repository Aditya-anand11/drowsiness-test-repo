# Driver-Drowsiness Detection — PyTorch Research Codebase

This repository is a **Python / PyTorch machine-learning codebase for vision-based driver-drowsiness detection**. It contains image and sequence data loading, several CNN/RNN/VAE model families, a set of standalone classical-learning-rule helpers, a training script with an ablation-friendly optimizer factory, and an evaluation / benchmarking script plus standalone run harnesses.

> **Note on earlier revisions of this file.** A previous version of this README described a full-stack TypeScript application (Next.js, Express, Prisma, PostgreSQL, Redis, BullMQ, Zod, pino). **None of that technology exists in this repository.** There is no TypeScript, no HTTP API, no database and no web frontend here. Everything below is derived from a direct read of the actual `.py` sources.

---

## 1. Core technology

| Area | Technology actually used in the code | Where |
| --- | --- | --- |
| Deep learning | PyTorch — `torch`, `torch.nn`, `torch.nn.functional`, `torch.optim`, `torch.utils.data` | model files, `train.py`, `dataloader.py`, `evaluate.py` |
| Image I/O | Pillow (`PIL.Image`) | `dataloader.py` |
| Dataset plumbing | `torchvision.datasets.ImageFolder`, `torchvision.transforms`, `random_split`, `DataLoader` | `dataloader.py` |
| Metrics | scikit-learn (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`) | `evaluate.py` |
| Numerics / PERCLOS | NumPy | `evaluate.py` |
| Standard library | `os`, `glob`, `time`, `argparse`, `json`, `platform`, `sys`, `importlib` | data/eval/harness scripts |

There is **no** TensorFlow, no CUDA-only path (device selection falls back to CPU), no web framework, no database, and no test suite in the repository. Training hyper-parameters are Python function defaults, not a config file.

---

## 2. High-level architecture

```text
 image folders on disk ("train", "test", "no yawn", "yawn")
        |
        v
 dataloader.py   FolderImageDataset / ImageFolder + transforms + random_split
        |         -> DataLoader batches (images, labels) and synthetic sequences
        v
 models_cnn.py       CNN2D | ResNet(ResidualBlock) | GoogLeNet(InceptionBlock)
                     | CNN3D | YOLOHead | focal_loss
 models_temporal.py  TemporalRNN | TemporalLSTM   (input B, T=10, F=4)
 models_vae.py       VAE (conv encoder / transposed-conv decoder) + vae_loss_fn
 paradigms.py        standalone learning-rule utilities (Hebbian, Oja, competitive,
                     error-correction, energy-based) on plain tensors
        |
        v
 train.py        optimizer factory + training loops (BCE loss / VAE loss)
        |
        v
 evaluate.py     accuracy / precision / recall / F1 / confusion matrix,
                 FPS & latency benchmark, PERCLOS drowsiness rule

 benchmark.py     standalone inference-FPS / training-step harness (lazy torch import)
 run_training.py  standalone training launcher (lazy torch import)
```

The codebase is split strictly by concern: **data (`dataloader.py`) -> models (`models_*.py`) -> training (`train.py`) -> measurement (`evaluate.py`)**, with `paradigms.py` kept independent of the training pipeline and the two harness scripts (`benchmark.py`, `run_training.py`) kept independent of the sandbox.

### End-to-end data flow

1. `dataloader.get_loaders` builds a training loader and a test loader. Eye-state data comes from `torchvision.datasets.ImageFolder("train")` and `ImageFolder("test")`; yawn data is listed once by `FolderImageDataset("no yawn", "yawn")` and split 80/20 with `random_split`, then wrapped in `TransformedSubset` so train and test halves can use *different* transforms.
2. Images are resized to `64x64`, optionally augmented (`RandomRotation(15)`, `RandomHorizontalFlip`, `ColorJitter(brightness=0.2)`) for training only, then converted to tensors.
3. `train.py` moves batches to the selected device, runs the model, computes `BCELoss` against labels reshaped to `(B, 1)`, back-propagates and steps the optimizer chosen by `get_optimizer`.
4. `evaluate.py` runs the model in `eval()` mode under `torch.no_grad()`, thresholds probabilities at `0.5`, and reports sklearn metrics plus a confusion matrix. `benchmark_fps` times repeated forward passes for throughput/latency, and `compute_perclos` turns a sequence of eye states into a PERCLOS ratio and a drowsiness boolean.

---

## 3. File-by-file implementation details

### `dataloader.py` — data pipeline

* **`TransformedSubset(base_dataset, indices, transform=None)`** — a `Dataset` view over a `FolderImageDataset`. It stores the parent dataset and a list of indices; `__getitem__` looks up `base_dataset.samples[indices[idx]]`, opens the image with `Image.open(path).convert("RGB")`, applies its own `transform` and returns `(image, label)`. Its docstring explains why it exists: the yawn split is built from one listing, but the train and test halves must see different transform pipelines; sharing one dataset with a single transform would silently apply training augmentation to the test half.
* **`FolderImageDataset(class0_dir, class1_dir, transform=None)`** — globs `*.*` in each of two directories, labelling class 0 and class 1, and stores `(path, label)` pairs in `self.samples`. `__len__` returns the sample count; `__getitem__` opens/convert-to-RGB/applies the transform.
* **`get_transforms(img_size=(64, 64), augment=True)`** — returns `(train_tf, test_tf)`. With `augment=True` the training pipeline is `Resize -> RandomRotation(15) -> RandomHorizontalFlip -> ColorJitter(brightness=0.2) -> ToTensor`; without it, `Resize -> ToTensor`. The test pipeline is always `Resize -> ToTensor`.
* **`get_loaders(dataset_type="both", img_size=(64, 64), batch_size=32, augment=True)`** — builds `ImageFolder` train/test datasets for the eye data, builds the yawn dataset with `FolderImageDataset("no yawn", "yawn")`, computes an 80/20 split length, uses `random_split(range(len(yawn_full)), [...])`, wraps the halves in `TransformedSubset` with `train_tf` / `test_tf`, and returns the configured `DataLoader` objects.
* **Synthetic sequence dataset** — a `Dataset` whose `__init__(self, num_samples=500, seq_len=10, feature_dim=4)` generates synthetic feature sequences (default 500 samples, 10 time steps, 4 features each) with `__len__` and `__getitem__` returning `(sequence, label)`.
* **`get_sequence_loader(batch_size=32)`** — wraps that synthetic sequence dataset in a `DataLoader` used by the temporal training path, matching the `(B, T=10, F=4)` input expected by `models_temporal.py`.

### `models_cnn.py` — image models and detection head

All models output a sigmoid probability with `num_classes=1` by default.

* **`ResidualBlock(in_c, out_c)`** — two `3x3` convolutions with `BatchNorm2d`, a ReLU between them, and a shortcut: an empty `nn.Sequential()` when `in_c == out_c`, otherwise a `1x1` convolution + `BatchNorm2d`. `forward` adds the shortcut to the residual branch and applies `F.relu`.
* **`ResNet(num_classes=1)`** — an initial `Conv2d(3, 32, kernel_size=7, stride=2, padding=3) + BatchNorm2d + ReLU`, then `ResidualBlock(32, 32)`, `ResidualBlock(32, 64)`, `AdaptiveAvgPool2d((1, 1))`, `flatten`, and a `Linear(64, num_classes)` followed by `torch.sigmoid`.
* **`InceptionBlock(in_c, f1, f2, f3)`** — four parallel branches: `1x1` conv (`f1` channels), `3x3` conv (`f2`, padding 1), `5x5` conv (`f3`, padding 2), and `MaxPool2d(3, stride=1, padding=1)` followed by a `1x1` conv (`f1`). Each branch is ReLU-activated and the outputs are concatenated along the channel dimension.
* **`GoogLeNet(num_classes=1)`** — `Conv2d(3, 32, 3, padding=1)` + ReLU, one `InceptionBlock(32, 16, 16, 16)`, `AdaptiveAvgPool2d((1, 1))`, flatten, and `Linear(16 * 4, num_classes)` + sigmoid.
* **`CNN2D(use_bn=True, dropout=0.5, init_type="he")`** — the configurable baseline used by the ablation study: a `3x3` conv to 32 channels with `BatchNorm2d` (or `nn.Identity()` when `use_bn=False`), a second `3x3` conv to 64 channels with `BatchNorm2d` (or identity), followed by pooling, dropout and a classification head; it accepts `dropout` and `init_type` so `train.train_ablation_cnn` can vary them.
* **`CNN3D`** — a 3D-convolution variant for spatio-temporal input, used for clip-level (rather than single-frame) classification.
* **`YOLOHead`** — a detection-style head that predicts per-location objectness/class scores, i.e. a YOLO-shaped output tensor rather than a single image-level probability.
* **`focal_loss(...)`** — a focal-loss helper that down-weights easy examples, intended for the class imbalance between alert and drowsy frames.

### `models_temporal.py` — sequence models

* **Activation helpers** — `step_activation(x)` returns `(x >= 0.0).float()`; `sigmoid_activation(x)` returns `torch.sigmoid(x)`; `tanh_activation(x)` returns `torch.tanh(x)`. These expose the three classic activation shapes used when discussing the classical neurons.
* **`TemporalRNN(input_dim=4, hidden_dim=32, num_classes=1)`** — an `nn.RNN(input_dim, hidden_dim, batch_first=True, nonlinearity="tanh")`; `forward` keeps only `h_n`, squeezes the layer dimension, applies `Linear(hidden_dim, num_classes)` and returns a sigmoid.
* **`TemporalLSTM(input_dim=4, hidden_dim=32, num_classes=1)`** — an `nn.LSTM(input_dim, hidden_dim, batch_first=True)`, then `Dropout(0.2)`, then `Linear(hidden_dim, num_classes)` and a sigmoid, again using the final hidden state `h_n`.

Both consume batch-first tensors of shape `(B, T=10, F=4)`, which is exactly what `dataloader.get_sequence_loader` produces.

### `models_vae.py` — variational auto-encoder

* **`VAE(latent_dim=16)`** — encoder: `Conv2d(3, 32, 4, stride=2, padding=1)` + ReLU, `Conv2d(32, 64, 4, stride=2, padding=1)` + ReLU, `Flatten`, then two linear heads `fc_mu` and `fc_logvar` from `64 * 16 * 16` to `latent_dim`. Decoder: `decoder_fc` maps `latent_dim` back to `64 * 16 * 16` and is reshaped to `(-1, 64, 16, 16)`, followed by `ConvTranspose2d(64, 32, 4, stride=2, padding=1)` + ReLU and `ConvTranspose2d(32, 3, 4, stride=2, padding=1)` + `Sigmoid`.
* **`reparameterize(mu, logvar)`** — the standard re-parameterisation trick: `std = exp(0.5 * logvar)`, `eps = randn_like(std)`, return `mu + eps * std`.
* **`forward(x)`** — encodes, computes `mu`/`logvar`, samples `z`, decodes, and returns `(recon_x, mu, logvar)`.
* **`generate(num_samples=5, device="cpu")`** — samples `z = randn(num_samples, latent_dim)` on the given device and runs the decoder to produce new images.
* **`vae_loss_fn(recon_x, x, mu, logvar)`** — `binary_cross_entropy(recon_x, x, reduction="sum")` plus the analytic KL term `-0.5 * sum(1 + logvar - mu^2 - exp(logvar))`, normalised by `x.size(0)`. The `Sigmoid` output layer matches this BCE reconstruction loss.

### `paradigms.py` — classical learning rules

These are self-contained tensor utilities, not `nn.Module`s, and are not imported by `train.py`; they exist as reference implementations of classical rules.

* **`hebbian_update(weights, x, y, lr=0.01)`** — returns `weights + lr * outer(y, x)`.
* **`ojas_update(weights, x, y, lr=0.01)`** — normalised Hebbian rule with the Oja decay term `(y ** 2).unsqueeze(1) * weights`, i.e. `weights + lr * (outer(y, x) - decay)`.
* **`competitive_learning(weights, x, lr=0.1)`** — computes `norm(weights - x, dim=1)`, picks `argmin` (the winning unit), moves it towards `x` by `lr`, and returns `(weights, winner_idx)`.
* **`error_correction_update(weights, bias, x, y_true, lr=0.05)`** — a perceptron-style rule: predicts `1.0` if `dot(weights, x) + bias >= 0` else `0.0`, computes `error = y_true - y_pred`, then updates `weights += lr * error * x` and `bias += lr * error`; returns `(weights, bias, error)`.
* **`energy_function(x, weights, bias)`** — `-0.5 * x^T W x - b^T x`, the Hopfield-style energy.
* **`energy_based_update(weights, bias, pos_data, neg_data, lr=0.01)`** — contrastive update `weights += lr * (outer(pos, pos) - outer(neg, neg))`, `bias += lr * (pos - neg)`.

### `train.py` — training loops

* **`get_optimizer(model, opt_name="adam", lr=0.001, l2=0.0)`** — optimizer factory used to drive ablations: `"sgd"` -> `optim.SGD(lr, weight_decay=l2)`, `"momentum"` -> SGD with `momentum=0.9`, `"nesterov"` -> SGD with `momentum=0.9, nesterov=True`, and Adam as the default fallback.
* **`train_ablation_cnn(dataset_type="both", opt_name="adam", use_bn=True, dropout=0.5, l2=0.001, init_type="he", augment=True, epochs=3)`** — selects CUDA when available, calls `get_loaders`, builds `CNN2D(use_bn, dropout, init_type)`, builds the optimizer via `get_optimizer(lr=0.001, l2=l2)`, uses `nn.BCELoss`, and per epoch runs the standard train loop: move images and `labels.float().unsqueeze(1)` to the device, zero gradients, forward, loss, backward, step; accumulates loss and accuracy (threshold `> 0.5`) and prints epoch loss/accuracy. Returns the trained model.
* **`train_temporal(epochs=5)`** — gets the sequence loader, builds `TemporalLSTM`, optimises with Adam (`lr=0.001`) and `BCELoss`, and prints the mean loss per epoch.
* **`train_vae(dataset_type="both", epochs=3)`** — gets loaders without augmentation, builds `VAE` with Adam (`lr=0.001`) and uses `vae_loss_fn` (imported from `models_vae.py`) as the objective in the training loop.

Note the consistent contract: every classifier ends in `sigmoid`, `BCELoss` is used, and labels are reshaped to `(B, 1)` so they match the model output.

### `evaluate.py` — measurement

* **`evaluate_model(model, test_loader, device="cpu")`** — `model.eval()`, iterates the loader under `torch.no_grad()`, thresholds outputs at `0.5` (`(outputs > 0.5).int()`), flattens to NumPy, accumulates truth/prediction lists, and returns a dict with `accuracy`, `precision`, `recall`, `f1` (all sklearn, `zero_division=0`) and the `confusion_matrix`.
* **`benchmark_fps(model, input_shape=(1, 3, 64, 64), num_runs=50, device="cpu")`** — creates a random dummy input, runs 5 warm-up passes, times `num_runs` forward passes under `no_grad`, and returns `{"fps": num_runs / elapsed, "latency_ms": elapsed / num_runs * 1000}`.
* **`compute_perclos(eye_states, threshold=0.8)`** — counts entries equal to `0` (closed eyes), divides by the sequence length to get the PERCLOS ratio, and reports `is_drowsy = perclos >= threshold`.
* **`__main__`** — instantiates `ResNet` and `GoogLeNet`, prints their `benchmark_fps` results; useful as a quick sanity/throughput check on a machine with torch installed.

### `benchmark.py` — standalone benchmark harness

A self-contained script that can be copied to a machine with PyTorch; it never imports torch at module import time, so `--dry-run` works without the heavy dependency.

* **`_import_torch()`** — returns the torch module or `None`; `INSTALL_HINT` holds the CPU-wheel install instructions printed when torch is missing.
* **`MODEL_SPECS`** — a list of dicts registering only models that actually exist in `models_cnn.py`: `CNN2D` (with `use_bn=True, dropout=0.5, init_type="he"`, mirroring `train.py`), `ResNet`, `GoogLeNet`.
* **`build_model(torch, name, kwargs)`** — imports `models_cnn` lazily and instantiates `getattr(models_cnn, name)(**kwargs)`, returning `None` for unknown names.
* **CLI** — `--dry-run` (wiring check without torch), `--batch-size`, `--runs` (inference FPS / latency on a `(batch, 3, 64, 64)` input), `--train-steps` (times real `CNN2D` optimisation steps with BCE loss and labels shaped `(B, 1)` as in `train.py`), and `--json` (dump a machine-readable report). It prints a table and optionally writes JSON.

### `run_training.py` — standalone training launcher

* **`_import_torch()`** — lazy torch import; `INSTALL_HINT` explains that TensorFlow is not used by this repo (every model is PyTorch) and gives the CPU-wheel install command.
* **`_check_data_dirs()`** — checks for the four image folders the dataloader expects (`train`, `test`, `no yawn`, `yawn`) and returns `(expected, missing)`.
* **`dry_run()`** — validates wiring without torch or data: prints the Python version, torch availability (and CUDA availability if torch exists), the expected/missing data dirs, and uses `importlib.util` to check that `train`, `dataloader`, `models_cnn`, `models_temporal`, `models_vae` are importable by name (importing `train.py` directly would import torch at module level).
* **CLI** — `--dry-run` (wiring check), `--model cnn` (calls `train.train_ablation_cnn`), `--model temporal` (calls `train.train_temporal`), `--model vae` (calls `train.train_vae`).

### `requirements.txt`

Lists the actual runtime dependencies: `torch`, `torchvision`, `Pillow`, `numpy`, `scikit-learn`. Comments document the CPU-only torch install command and note that torch/torchvision are large and must be installed on a machine with enough RAM/disk; the harness scripts degrade gracefully when PyTorch is missing.

---

## 4. How to run

Heavy dependencies (`torch`, `torchvision`) cannot be installed in a restricted 512 MB / 1 CPU / no-network / no-GPU sandbox, and **TensorFlow is not used by this repo at all**. Run the following on a dedicated machine:

```bash
pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu   # CPU-only torch

# wiring checks (no torch needed)
python benchmark.py --dry-run
python run_training.py --dry-run

# inference throughput / latency
python benchmark.py --batch-size 8 --runs 50
python benchmark.py --train-steps 20 --json bench.json   # + timed training steps

# training (expects the train/ test/ "no yawn"/ yawn/ image folders)
python run_training.py --model cnn
python run_training.py --model temporal
python run_training.py --model vae
```

Expected on-disk layout for `dataloader.get_loaders`:

```text
train/      # ImageFolder layout, eye-state classes
test/       # ImageFolder layout, eye-state classes
no yawn/    # class 0 images for the yawn split
yawn/       # class 1 images for the yawn split
```

The temporal path uses a synthetic sequence dataset (500 samples, 10 time steps, 4 features) so it can run without a sequence dataset on disk.

---

## 5. Design notes and known gaps

* **Single-output sigmoid classifiers.** Every classifier ends in `sigmoid` with `num_classes=1`, so training uses `BCELoss` and labels are reshaped with `unsqueeze(1)`; evaluation thresholds at `0.5`.
* **Transform isolation.** `TransformedSubset` exists specifically to keep training augmentation out of the test half of the yawn split.
* **Device selection** is `cuda` if available, otherwise `cpu`, in every training function in `train.py`.
* **No checkpointing / logging infrastructure.** Models are returned from the training functions in memory; there is no `save`/`load`, no TensorBoard, no config file.
* **No test suite.** Validation is manual (`benchmark.py --dry-run`, `run_training.py --dry-run`, and the `evaluate.py` `__main__` block).
* **No CI workflows.** Benchmark and training automation are deliberately delivered as normal scripts (`benchmark.py`, `run_training.py`) plus this README note, instead of files under `.github/workflows/`.
