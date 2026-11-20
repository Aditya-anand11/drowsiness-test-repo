# Architecture and Code-Level Functionality Guide

This document explains the core technology and the functionality of each source file in this repository, based on a direct read of the code.

## 1. What this repository actually is

The repository is a **Python / PyTorch machine-learning codebase for driver-drowsiness detection**. It contains data loading, several model families (CNN, temporal RNN/LSTM, VAE), classical unsupervised/learning-rule helpers, a training script, and an evaluation/benchmarking script.

> **Important discrepancy:** `readme.md` describes a "full-stack TypeScript application" (Next.js, Express, Prisma, PostgreSQL, Redis, BullMQ, Zod, pino). **None of that technology exists in this repository.** There is no TypeScript, no API server, no database, and no frontend here. The README appears to belong to a different project and should not be treated as documentation of this code. Everything below is derived from the actual `.py` files.

## 2. Core technology

| Area | Technology present in the code |
| --- | --- |
| Deep learning | PyTorch (`torch`, `torch.nn`, `torch.nn.functional`, `torch.optim`, `torch.utils.data`) |
| Image handling | Pillow (`PIL.Image`), `torchvision.datasets.ImageFolder`, `torchvision.transforms` |
| Metrics | scikit-learn (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`) |
| Numerics | NumPy |
| Data plumbing | Python `glob`, `os` |

## 3. High-level architecture

```text
 image folders on disk ("train", "test", "no yawn", "yawn")
        |
        v
dataloader.py  -- FolderImageDataset / ImageFolder + transforms + random_split
        |            -> DataLoader batches (images, labels) and synthetic sequences
        v
models_*.py    -- CNN2D / ResNet / GoogLeNet | TemporalRNN / TemporalLSTM | VAE
        |
        v
train.py       -- optimizer factory + training loops (BCE loss / VAE loss)
        |
        v
evaluate.py    -- metrics, confusion matrix, FPS/latency benchmark, PERCLOS drowsiness rule

paradigms.py   -- standalone learning-rule utilities (Hebbian, Oja, competitive,
                  error-correction, energy-based) operating on plain tensors
```

The pipeline is split by concern: **data (`dataloader.py`) -> models (`models_*.py`) -> training (`train.py`) -> measurement (`evaluate.py`)**, with `paradigms.py` kept separate as a library of low-level weight-update rules.

## 4. File-by-file functionality

### `dataloader.py` - data ingestion

- **`FolderImageDataset`** - generic two-folder binary dataset. The constructor takes `class0_dir` and `class1_dir`, globs `*.*` in each, and stores `(path, label)` pairs with label `0` for the first folder and `1` for the second. `__getitem__` opens the image with Pillow, forces `RGB`, applies the transform, and returns `(image, label)`.
- **`get_transforms(img_size=(64, 64), augment=True)`** - returns a `(train_tf, test_tf)` pair. When `augment` is true the training pipeline is `Resize -> RandomRotation(15) -> RandomHorizontalFlip -> ColorJitter(brightness=0.2) -> ToTensor`; otherwise it is just `Resize -> ToTensor`. The test pipeline is always `Resize -> ToTensor` (no augmentation).
- **`get_loaders(dataset_type="both", img_size=(64, 64), batch_size=32, augment=True)`** - builds an eye-state dataset with `datasets.ImageFolder("train", ...)` and `ImageFolder("test", ...)` (fixed relative paths), and a yawn dataset through `FolderImageDataset("no yawn", "yawn", ...)`. The yawn set is split 80/20 with `random_split`. `dataset_type` selects `"eyes"`, `"yawn"`, or (default) both merged with `ConcatDataset`. Returns shuffled training and non-shuffled test `DataLoader`s.
- **`SequenceDataset`** - synthetic temporal dataset: `num_samples x seq_len x feature_dim` (default 500 x 10 x 4) random tensors; the label is `1.0` when the mean of feature 0 across the sequence is positive, else `0.0`, returned as shape `(1,)`.
- **`get_sequence_loader(batch_size=32)`** - wraps `SequenceDataset` in a shuffled `DataLoader`.

Note: because the paths are hard-coded relative strings, the scripts must be run from a working directory that contains `train/`, `test/`, `no yawn/`, and `yawn/` folders.

### `models_cnn.py` - convolutional models

- **`ResidualBlock(in_c, out_c)`** - two 3x3 convolutions with BatchNorm; a 1x1 conv + BatchNorm shortcut is inserted only when `in_c != out_c`. `forward` computes `res = shortcut(x)`, applies `relu(bn1(conv1(x)))`, then `bn2(conv2(...))`, and returns `relu(out + res)`.
- **`ResNet(num_classes=1)`** - a compact residual net: stem `Conv2d(3, 32, kernel 7, stride 2, padding 3) -> BatchNorm -> ReLU`, then `ResidualBlock(32, 32)` and `ResidualBlock(32, 64)`, `AdaptiveAvgPool2d((1, 1))`, `Flatten`, `Linear(64, num_classes)`, and `sigmoid` on the output (so it produces a single probability, suitable for binary `BCELoss`).
- **`InceptionBlock(in_c, f1, f2, f3)`** - four parallel branches: 1x1 conv (`f1`), 3x3 conv (`f2`), 5x5 conv (`f3`), and max-pool-then-1x1 conv (`f1`); each branch is ReLU-activated and the four outputs are concatenated on the channel axis.
- **`GoogLeNet(num_classes=1)`** - a minimal inception-style network: `Conv2d(3, 32, 3)` + ReLU, one `InceptionBlock(32, 16, 16, 16)` (hence `16 * 4 = 64` channels out), `AdaptiveAvgPool2d`, `Flatten`, `Linear(64, num_classes)`, `sigmoid`.
- **`CNN2D(use_bn=True, dropout=0.5, init_type="he")`** - the configurable ablation model used by `train.py`. The portion visible in the source shows `conv1: Conv2d(3, 32, 3, padding 1)`, `bn1 = BatchNorm2d(32)` when `use_bn` else `nn.Identity()`, `conv2: Conv2d(32, 64, 3, padding 1)`, `bn2 = BatchNorm2d(64)` when `use_bn` else `nn.Identity()`, and `self.drop = nn.Dropout(dropout)` when `dropout > 0` else `nn.Identity()`. Its remaining layers and the exact `forward`/weight-initialization (`init_type`, e.g. He) code were not visible in the read of the file, so they are not described here.

### `models_temporal.py` - sequence models

- **`step_activation`, `sigmoid_activation`, `tanh_activation`** - small helper functions implementing a step function (`x >= 0 -> 1.0`), sigmoid, and tanh.
- **`TemporalRNN(input_dim=4, hidden_dim=32, num_classes=1)`** - a `nn.RNN` with `batch_first=True` and `tanh` nonlinearity; it keeps only the final hidden state `h_n`, squeezes the layer dimension, passes it through `Linear(hidden_dim, num_classes)` and `sigmoid`. This consumes the 4-feature vectors produced by `SequenceDataset`.
- **`TemporalLSTM(input_dim=4, hidden_dim=32, num_classes=1)`** - same contract but with `nn.LSTM`, plus `Dropout(0.2)` on the final hidden state before the linear layer and `sigmoid` output.

### `models_vae.py` - variational autoencoder

- **`VAE(latent_dim=16)`** - convolutional encoder (`Conv2d(3,32,4,s2,p1) -> ReLU -> Conv2d(32,64,4,s2,p1) -> ReLU -> Flatten`) feeding two heads, `fc_mu` and `fc_logvar`, each `Linear(64*16*16, latent_dim)`. The decoder mirrors it: `decoder_fc: Linear(latent_dim, 64*16*16)`, reshape to `(-1, 64, 16, 16)`, then two `ConvTranspose2d` layers (64->32, 32->3) with ReLU and a final `Sigmoid`.
- **`reparameterize(mu, logvar)`** - the standard reparameterization trick: `std = exp(0.5 * logvar)`, `eps ~ N(0, 1)`, return `mu + eps * std`.
- **`forward(x)`** - encodes, samples `z`, decodes, and returns `(recon_x, mu, logvar)`. It implies a 64x64 input because the encoder's flatten size is fixed at `64*16*16`.
- **`generate(num_samples=5, device="cpu")`** - samples `z ~ N(0, 1)` of size `fc_mu.out_features`, decodes it, and returns generated images.
- **`vae_loss_fn(recon_x, x, mu, logvar)`** - binary-cross-entropy reconstruction term (`reduction="sum"`) plus the analytic KL divergence `-0.5 * sum(1 + logvar - mu^2 - exp(logvar))`, normalized by batch size.

### `paradigms.py` - learning-rule utilities

Standalone functions that update plain weight tensors (not `nn.Module`s), illustrating classical learning paradigms:

- **`hebbian_update(weights, x, y, lr)`** - `weights + lr * outer(y, x)`.
- **`ojas_update(weights, x, y, lr)`** - Hebbian term minus Oja's decay term `(y^2) * weights`, which keeps weight norms bounded.
- **`competitive_learning(weights, x, lr)`** - finds the prototype with the smallest Euclidean distance (`argmin` of `norm(weights - x)`) and moves only that winner toward `x`; returns `(weights, winner_idx)`.
- **`error_correction_update(weights, bias, x, y_true, lr)`** - a perceptron-style rule: predict `1.0` if `dot(weights, x) + bias >= 0` else `0.0`, compute `error = y_true - y_pred`, then update `weights += lr * error * x` and `bias += lr * error`; returns `(weights, bias, error)`.
- **`energy_function(x, weights, bias)`** - Hopfield-style energy `-0.5 * x^T W x - b^T x`.
- **`energy_based_update(weights, bias, pos_data, neg_data, lr)`** - contrastive (positive vs. negative phase) update of the outer-product weights and the bias.

Several of these mutate tensors in place (`weights[winner_idx] += ...`, `weights += ...`), so callers must pass tensors that are safe to modify.

### `train.py` - training entry points

- **`get_optimizer(model, opt_name, lr, l2)`** - optimizer factory mapping `"sgd"` -> `SGD`, `"momentum"` -> `SGD(momentum=0.9)`, `"nesterov"` -> `SGD(momentum=0.9, nesterov=True)`, anything else -> `Adam`. `l2` is passed as `weight_decay` in every case, which is how this codebase performs L2 regularization.
- **`train_ablation_cnn(dataset_type, opt_name, use_bn, dropout, l2, init_type, augment, epochs=3)`** - device selection (`cuda` if available), loads data via `get_loaders`, builds `CNN2D(use_bn, dropout, init_type)`, uses `BCELoss`; each epoch runs forward pass, loss, backward, step, and reports mean loss plus thresholded (`> 0.5`) accuracy. Returns the trained model. This is the ablation harness: BatchNorm on/off, dropout rate, optimizer, L2, initialization, and augmentation can each be varied.
- **`train_temporal(epochs=5)`** - same loop shape for `TemporalLSTM` on `get_sequence_loader()` data with Adam and `BCELoss`.
- **`train_vae(dataset_type="both", epochs=3)`** - builds `VAE` and an Adam optimizer over `get_loaders(..., augment=False)`. In the version read, the file ends inside the training epoch loop (after `model.train()`), so the reconstruction/KL optimization step and any `main`/CLI entry point are not present in the file as read.

### `evaluate.py` - evaluation and benchmarking

- **`evaluate_model(model, test_loader, device="cpu")`** - runs the model in `eval()` mode under `torch.no_grad()`, thresholds outputs at `0.5`, collects predictions and labels, and returns a dict with `accuracy`, `precision`, `recall`, `f1` (all scikit-learn, `zero_division=0`) and the `confusion_matrix`.
- **`benchmark_fps(model, input_shape=(1, 3, 64, 64), num_runs=50, device="cpu")`** - warms up with 5 forward passes, times `num_runs` passes with `time.time()`, and returns `fps` and `latency_ms`. This is the speed/latency comparison used for the architecture trade-off (e.g. ResNet vs. GoogLeNet).
- **`compute_perclos(eye_states, threshold=0.8)`** - the domain rule for drowsiness: counts frames whose eye state equals `0` (closed), divides by the number of frames to obtain PERCLOS, and flags drowsiness when `perclos >= threshold`. Returns `(perclos, is_drowsy)`.
- **`__main__` block** - instantiates `ResNet()` and `GoogLeNet()` and prints their benchmark results, so `python evaluate.py` performs a quick speed comparison.

## 5. End-to-end flow

1. Place the image folders (`train`, `test`, `no yawn`, `yawn`) next to the scripts.
2. `get_loaders` builds augmented, batched 64x64 RGB tensors with binary labels (eye open/closed, or no-yawn/yawn), optionally concatenated into one task.
3. `train_ablation_cnn` trains `CNN2D` with a settable optimizer/regularization/initialization configuration; `train_temporal` trains the LSTM on synthetic 10-step sequences; `train_vae` is set up to train the convolutional VAE.
4. `evaluate_model` produces accuracy/precision/recall/F1 and a confusion matrix on the test split; `benchmark_fps` compares inference speed; `compute_perclos` turns a stream of per-frame eye states into a drowsiness decision.

## 6. Cross-file contracts to respect when extending

- Every model feeding `BCELoss` (`CNN2D`, `ResNet`, `GoogLeNet`, `TemporalRNN`, `TemporalLSTM`) already ends in `sigmoid` and outputs one value per sample, so labels are expected as float shape `(N, 1)` (as `train_ablation_cnn` does with `labels.float().unsqueeze(1)`).
- `SequenceDataset` emits 4 features per timestep; `TemporalRNN` and `TemporalLSTM` default to `input_dim=4` - changing one requires changing the other.
- `VAE`'s flattened encoder size `64*16*16` is hard-coded on both sides, so `img_size` used for VAE data must stay 64x64 unless the linear layers are also recomputed.
- `evaluate_model` and `benchmark_fps` assume single-logit sigmoid outputs at 64x64 resolution; they will not work unchanged for a multi-class classifier.
