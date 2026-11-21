# Architecture and Code-Level Functionality Guide

This document explains the core technology and the functionality of each source file in this repository. It is based on a direct read of the actual `.py` sources, not on the project README.

## 1. What this repository actually is

The repository is a **Python / PyTorch machine-learning codebase for vision-based driver-drowsiness detection**. It contains image/sequence data loading, several model families (2D CNN, residual CNN, an Inception-style CNN, a 3D CNN, a YOLO-style head, temporal RNN/LSTM, and a VAE), a library of classical learning rules, a training script with an ablation-friendly optimizer factory, and an evaluation/benchmarking script.

> **Important discrepancy:** `readme.md` (as it was originally) described a "full-stack TypeScript application" (Next.js, Express, Prisma, PostgreSQL, Redis, BullMQ, Zod, pino). **None of that technology exists in this repository.** There is no TypeScript, no API server, no database and no web frontend here. Everything below is derived from the actual `.py` files.

## 2. Core technology

| Area | Technology present in the code | Where |
| --- | --- | --- |
| Deep learning | PyTorch (`torch`, `torch.nn`, `torch.nn.functional`, `torch.optim`, `torch.utils.data`) | all model files, `train.py` |
| Image handling | Pillow (`PIL.Image`), `torchvision.datasets.ImageFolder`, `torchvision.transforms` | `dataloader.py` |
| Metrics | scikit-learn (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`) | `evaluate.py` |
| Numerics / PERCLOS | NumPy | `evaluate.py` |
| Data plumbing | Python `glob`, `os`, `time` | `dataloader.py`, `evaluate.py` |

There is **no** `requirements.txt`, **no** test suite, **no** configuration file, and **no** model-checkpoint / logging infrastructure in the repository. Training hyperparameters are Python function defaults.

## 3. High-level architecture

```text
 image folders on disk ("train", "test", "no yawn", "yawn")
        |
        v
dataloader.py  -- FolderImageDataset / ImageFolder + transforms + random_split
        |            -> DataLoader batches (images, labels) and synthetic sequences
        v
models_cnn.py       -- CNN2D | ResNet (ResidualBlock) | GoogLeNet (InceptionBlock)
                       | CNN3D | YOLOHead | focal_loss
train.py            -- optimizer factory + training loops (BCE loss / VAE loss)
        v
evaluate.py         -- metrics, confusion matrix, FPS/latency benchmark, PERCLOS rule

models_temporal.py  -- TemporalRNN | TemporalLSTM (input B, T=10, F=4)
models_vae.py       -- VAE (conv encoder / transposed-conv decoder) + vae_loss_fn
paradigms.py        -- standalone learning-rule utilities (Hebbian, Oja, competitive,
                       error-correction, energy-based) operating on plain tensors
```

The pipeline is split by concern: **data (`dataloader.py`) -> models (`models_*.py`) -> training (`train.py`) -> measurement (`evaluate.py`)**, with `paradigms.py` kept separate as a library of low-level weight-update rules.

## 4. File-by-file functionality

### `dataloader.py` - data ingestion

- **`FolderImageDataset(Dataset)`** - generic two-folder binary dataset. The constructor takes `class0_dir` and `class1_dir`, globs `*.*` in each folder, and stores `(path, label)` pairs with label `0` for the first folder and label `1` for the second. `__len__` returns the number of samples. `__getitem__` opens the image with Pillow, converts it to `RGB`, applies the transform (if any) and returns `(image, label)`.
- **`get_transforms(img_size=(64, 64), augment=True)`** - returns `(train_tf, test_tf)`. The train transform resizes to `img_size`, applies `RandomRotation(15)`, `RandomHorizontalFlip()`, `ColorJitter(brightness=0.2)` and `ToTensor()`. When `augment=False`, the train transform only resizes and converts to tensor. The test transform always just resizes and converts to tensor.
- **`get_loaders(dataset_type="both", img_size=(64, 64), batch_size=32, augment=True)`** - builds an `ImageFolder` for `"train"` and `"test"` (the eye open/closed data), and a `FolderImageDataset("no yawn", "yawn")` for yawn data, which is split 80/20 into train/test with `random_split`. `dataset_type` selects `"eyes"`, `"yawn"`, or `"both"` (the default concatenates both train sets and both test sets with `ConcatDataset`). It returns two shuffled (train) / non-shuffled (test) `DataLoader` objects with `batch_size`.
- **`SequenceDataset(Dataset)`** - synthetic temporal dataset. It generates `X` of shape `(num_samples, seq_len=10, feature_dim=4)` with `torch.randn`, and derives the label `y` from whether the mean of the first feature over time is positive, giving a `(num_samples, 1)` float target.
- **`get_sequence_loader(batch_size=32)`** - wraps a `SequenceDataset` in a shuffled `DataLoader`.

### `models_cnn.py` - convolutional models and losses

- **`ResidualBlock(in_c, out_c)`** - a residual unit with two 3x3 convolutions plus batch-norm (`conv1 -> bn1 -> ReLU -> conv2 -> bn2`). A 1x1 convolution + batch-norm `shortcut` is inserted only when `in_c != out_c`. The forward pass adds the shortcut to the residual branch and applies a final ReLU.
- **`ResNet(num_classes=1)`** - a small residual network: a stem `Conv2d(3, 32, kernel_size=7, stride=2, padding=3) + BatchNorm + ReLU`, followed by `ResidualBlock(32, 32)` and `ResidualBlock(32, 64)`, an `AdaptiveAvgPool2d((1, 1))`, a `Linear(64, num_classes)` and a `Sigmoid`. It outputs a single probability.
- **`InceptionBlock(in_c, f1, f2, f3)`** - a mini Inception module with four parallel branches: a 1x1 conv (f1), a 3x3 conv (f2), a 5x5 conv (f3) and a max-pool-3x3 followed by a 1x1 conv (f1). The branch outputs are concatenated along the channel dimension.
- **`GoogLeNet(num_classes=1)`** - a `Conv2d(3, 32, kernel_size=3, padding=1) + ReLU`, one `InceptionBlock(32, 16, 16, 16)`, an `AdaptiveAvgPool2d((1, 1))`, a `Linear(16 * 4, num_classes)` and a `Sigmoid`.
- **`CNN2D(use_bn=True, dropout=0.5, init_type="he")`** - the configurable baseline CNN used for ablations. It stacks `Conv2d(3, 32)` + optional `BatchNorm2d` (else `Identity`), `Conv2d(32, 64)` + optional BN, an optional `Dropout`, an `AdaptiveAvgPool2d((4, 4))`, `Linear(64 * 4 * 4, 32)` and `Linear(32, 1)`. Its `_init_weights(init_type)` iterates over `self.modules()` and initialises the weights according to the requested scheme. The forward pass runs both convs with 2x2 max-pooling after each, pools, flattens, applies dropout + ReLU on `fc1`, and returns `sigmoid(fc2(x))`.
- **`CNN3D(num_classes=1)`** - a volumetric (video-clip) variant that starts with `Conv3d(3, 16, kernel_size=(3, 3, 3), padding=1)` (as seen in the source), intended for spatio-temporal input.
- **`YOLOHead(in_c=64, num_classes=3, num_anchors=2)`** - a detection-style head parameterised by input channels, number of classes and number of anchors.
- **`focal_loss(preds, targets, alpha=0.25, gamma=2.0)`** - a focal-loss helper for class-imbalanced binary/one-hot targets.

### `models_temporal.py` - sequence models

The module also exposes three plain activation helpers: `step_activation(x)` (returns `(x >= 0.0)` as float), `sigmoid_activation(x)` and `tanh_activation(x)`.

- **`TemporalRNN(input_dim=4, hidden_dim=32, num_classes=1)`** - wraps `nn.RNN(input_dim, hidden_dim, batch_first=True, nonlinearity="tanh")`; the final hidden state is fed to a `Linear(hidden_dim, num_classes)` and a `Sigmoid`.
- **`TemporalLSTM(input_dim=4, hidden_dim=32, num_classes=1)`** - wraps `nn.LSTM(input_dim, hidden_dim, batch_first=True)`, applies `Dropout(0.2)` to the final hidden state, then `Linear(hidden_dim, num_classes)` and a `Sigmoid`. This is the model actually used by `train_temporal`.

### `models_vae.py` - variational autoencoder

- **`VAE(latent_dim=16)`** - a convolutional VAE. The `encoder` is `Conv2d(3, 32, kernel_size=4, stride=2, padding=1) -> ReLU -> Conv2d(32, 64, kernel_size=4, stride=2, padding=1) -> ReLU -> Flatten`, producing a `64 * 16 * 16` vector; `fc_mu` and `fc_logvar` map it to the latent dimension. `decoder_fc` maps the latent vector back to `64 * 16 * 16`, reshaped to `(-1, 64, 16, 16)` and passed through `ConvTranspose2d(64, 32, 4, stride=2, padding=1) -> ReLU -> ConvTranspose2d(32, 3, 4, stride=2, padding=1) -> Sigmoid`.
- **`reparameterize(mu, logvar)`** - the reparameterisation trick: `mu + eps * exp(0.5 * logvar)` with `eps ~ N(0, 1)`.
- **`forward(x)`** - encodes, samples `z`, decodes and returns `(recon_x, mu, logvar)`.
- **`generate(num_samples=5, device="cpu")`** - samples from the prior and decodes, i.e. unconditional image generation.
- **`vae_loss_fn(recon_x, x, mu, logvar)`** - returns `(BCE reconstruction + KL divergence) / batch_size`, where the KL term is `-0.5 * sum(1 + logvar - mu^2 - exp(logvar))`.

### `train.py` - training orchestration

- **`get_optimizer(model, opt_name="adam", lr=0.001, l2=0.0)`** - ablation-friendly optimizer factory. Supports `"sgd"`, `"momentum"` (SGD + momentum 0.9), `"nesterov"` (SGD + momentum 0.9 + `nesterov=True`) and defaults to Adam. All variants pass through `weight_decay=l2` (L2 regularisation).
- **`train_ablation_cnn(dataset_type="both", opt_name="adam", use_bn=True, dropout=0.5, l2=0.001, init_type="he", augment=True, epochs=3)`** - selects the device (CUDA if available), builds loaders via `get_loaders`, instantiates `CNN2D` with the ablation flags, builds the optimizer, and uses `nn.BCELoss`. For each epoch it iterates the train loader (labels reshaped with `.float().unsqueeze(1)`), zeroes gradients, runs forward/backward/step, accumulates loss, computes predictions via `outputs > 0.5`, and prints average loss and accuracy per epoch. Returns the trained model.
- **`train_temporal(epochs=5)`** - builds the synthetic `get_sequence_loader`, instantiates `TemporalLSTM`, uses Adam + `nn.BCELoss`, and prints average loss per epoch. Returns the model.
- **`train_vae(dataset_type="both", epochs=3)`** - builds image loaders (`augment=False`), instantiates `VAE`, uses Adam, and trains with `vae_loss_fn` (reconstruction + KL).

### `evaluate.py` - metrics and benchmarking

- **`evaluate_model(model, test_loader, device="cpu")`** - runs the model in `eval()` mode under `torch.no_grad()`, thresholds outputs at `0.5`, collects true/predicted labels and returns a dict with `accuracy`, `precision`, `recall`, `f1` and the `confusion_matrix` (scikit-learn metrics, `zero_division=0`).
- **`benchmark_fps(model, input_shape=(1, 3, 64, 64), num_runs=50, device="cpu")`** - warms up with 5 forward passes, times `num_runs` forward passes with `time.time()`, and returns `fps` and `latency_ms`.
- **`compute_perclos(eye_states, threshold=0.8)`** - computes the fraction of frames whose recorded eye state equals `0` (closed) and flags drowsiness when the PERCLOS ratio is `>= threshold`. Returns `(perclos, is_drowsy)`.
- **`__main__` block** - instantiates `ResNet` and `GoogLeNet` and prints their benchmarks.

### `paradigms.py` - classical learning-rule utilities

A standalone collection of tensor-level learning rules (no `nn.Module`), demonstrating classical update paradigms:

- **`hebbian_update(weights, x, y, lr=0.01)`** - `weights + lr * outer(y, x)`.
- **`ojas_update(weights, x, y, lr=0.01)`** - Oja's rule: Hebbian term minus a `y^2`-weighted decay term.
- **`competitive_learning(weights, x, lr=0.1)`** - finds the row of `weights` closest (L2) to `x`, updates that winning row toward `x`, and returns the winner index.
- **`error_correction_update(weights, bias, x, y_true, lr=0.05)`** - perceptron-style update: predicts 0/1 from the sign of `dot(weights, x) + bias`, then applies `w += lr * error * x` and `b += lr * error`.
- **`energy_function(x, weights, bias)`** - an energy/Hopfield-style scalar `-0.5 * x^T W x - b^T x`.
- **`energy_based_update(weights, bias, pos_data, neg_data, lr=0.01)`** - moves the energy surface by `lr * (outer(pos,pos) - outer(neg,neg))` for the weights and `lr * (pos - neg)` for the bias.

## 5. Data and label conventions

- Images are expected as folders: `train/` and `test/` with one subdirectory per eye class (used by `ImageFolder`), and `no yawn/` / `yawn/` directories for the yawn dataset (used by `FolderImageDataset`).
- The binary label convention is `0` = first class (e.g. eye open / no yawn) and `1` = second class (e.g. eye closed / yawn).
- All CNN models output a single sigmoid probability per image, so training uses `BCELoss` and evaluation thresholds at `0.5`.
- Temporal models consume `(batch, seq_len=10, feature_dim=4)` sequences and also output a single sigmoid probability.
- The VAE consumes RGB images resized to `(64, 64)` (its encoder assumes a spatial size that reduces to `16x16` after two stride-2 convolutions).

## 6. Execution flow

There is no CLI entry point wiring the whole pipeline. The scripts are imported/run individually:

1. `dataloader.get_loaders(...)` or `dataloader.get_sequence_loader()` produces batches.
2. `train.train_ablation_cnn(...)`, `train.train_temporal(...)` or `train.train_vae(...)` trains a model.
3. `evaluate.evaluate_model(...)` computes metrics on a held-out loader, `evaluate.benchmark_fps(...)` measures inference speed, and `evaluate.compute_perclos(...)` applies the PERCLOS drowsiness rule.
4. `python evaluate.py` runs the FPS benchmark for `ResNet` and `GoogLeNet`.

## 7. Known limitations observed in the code

- The README previously documented a TypeScript stack that is not present; it has been rewritten to match the code.
- There is no `requirements.txt`, so dependencies (`torch`, `torchvision`, `Pillow`, `scikit-learn`, `numpy`) must be installed manually.
- There is no test suite or CI.
- `get_loaders` hardcodes the relative folder names (`train`, `test`, `no yawn`, `yawn`) and will raise if those directories are absent.
- `VAE.generate` reads the latent size from `self.fc_mu.out_features`.
