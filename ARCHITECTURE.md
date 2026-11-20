# Architecture and Code-Level Functionality Guide

This document explains the core technology and the function-by-function behaviour of each source file in this repository. It is based on a direct read of the Python sources listed below.

## 1. What the repository is

The repository is a **PyTorch machine-learning codebase for driver-drowsiness detection**. It is split into four concerns:

- data ingestion and batching,
- model families (convolutional, temporal/recurrent, variational autoencoder),
- training loops and optimizer construction,
- evaluation, benchmarking and the PERCLOS drowsiness rule.

A separate module (`paradigms.py`) holds low-level, plain-tensor learning-rule utilities that are **not** wired into the training loops.

## 2. Core technology

| Area | Technology actually used in the code |
| --- | --- |
| Deep learning | PyTorch: `torch`, `torch.nn`, `torch.nn.functional`, `torch.optim`, `torch.utils.data` (`Dataset`, `DataLoader`, `random_split`, `ConcatDataset`) |
| Image input | Pillow (`PIL.Image`), `torchvision.datasets.ImageFolder`, `torchvision.transforms` |
| Metrics | scikit-learn: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix` |
| Numerics | NumPy |
| File plumbing | Python `os`, `glob` |

Runtime dependencies implied by the imports: `torch`, `torchvision`, `Pillow`, `numpy`, `scikit-learn`.

## 3. High-level architecture

```text
image folders on disk  ("train", "test", "no yawn", "yawn")
        |
        v
dataloader.py   FolderImageDataset / ImageFolder + transforms + random_split
        |       -> DataLoader over (image, label) batches
        |       -> get_sequence_loader: synthetic (sequence, label) batches
d        |
        v
models_cnn.py       CNN2D, ResNet, GoogLeNet, ResidualBlock, InceptionBlock
models_temporal.py  TemporalRNN, TemporalLSTM + activation helpers
models_vae.py       VAE and vae_loss_fn
        |
        v
train.py        get_optimizer + train_ablation_cnn / train_temporal / train_vae
        |
        v
evaluate.py     evaluate_model metrics, benchmark_fps, compute_perclos

paradigms.py    standalone weight-update rules (Hebbian, Oja, competitive,
                error correction, energy-based) on plain tensors
```

The data path is always **dataset -> DataLoader -> model -> (loss, optimizer)**, and every supervised model emits a **single sigmoid output** compared with a binary label through `BCELoss`.

## 4. File-by-file functionality

### `dataloader.py` - data ingestion and batching

- **`FolderImageDataset(Dataset)`** - generic two-folder binary image dataset.
  - Constructor takes `class0_dir`, `class1_dir` and an optional `transform`. It globs `*.*` in each directory and stores `(path, label)` pairs, label `0` for the first folder and `1` for the second.
  - `__len__` returns the number of stored samples.
  - `__getitem__` opens the file with `Image.open`, converts it to `RGB`, applies the transform if one was supplied, and returns `(image, label)`.
- **`get_transforms(img_size=(64, 64), augment=True)`** - returns a pair `(train_tf, test_tf)`.
  - With `augment=True` the training pipeline is `Resize(img_size) -> RandomRotation(15) -> RandomHorizontalFlip() -> ColorJitter(brightness=0.2) -> ToTensor()`.
  - With `augment=False` it is `Resize(img_size) -> ToTensor()`.
  - The test pipeline is always `Resize(img_size) -> ToTensor()` (no augmentation).
- **`get_loaders(dataset_type="both", img_size=(64, 64), batch_size=32, augment=True)`** - builds the train/test `DataLoader` pair.
  - Eyes data: `datasets.ImageFolder("train", ...)` and `datasets.ImageFolder("test", ...)`.
  - Yawn data: `FolderImageDataset("no yawn", "yawn", ...)`, split 80/20 with `random_split` into train and test subsets.
  - `dataset_type="eyes"` uses only the ImageFolder sets, `"yawn"` uses only the split yawn sets, anything else concatenates both with `ConcatDataset`.
  - Train loader shuffles; test loader does not.
- **`SequenceDataset(Dataset)`** - synthetic temporal data: `X = torch.randn(num_samples, seq_len, feature_dim)` with defaults `num_samples=500`, `seq_len=10`, `feature_dim=4`; the label is `(X[:, :, 0].mean(dim=1) > 0)` as a float of shape `(N, 1)`, i.e. the sign of the average of the first feature over time.
- **`get_sequence_loader(batch_size=32)`** - wraps `SequenceDataset` in a shuffled `DataLoader`.

### `models_cnn.py` - convolutional models

- **`ResidualBlock(nn.Module)`** - two `3x3` convolutions with batch norm (`bn1`, `bn2`). The shortcut is an empty `Sequential` when `in_c == out_c`, otherwise a `1x1` convolution plus batch norm for channel projection. `forward` computes `res = shortcut(x)`, `relu(bn1(conv1(x)))`, then `bn2(conv2(...))`, and returns `relu(out + res)`.
- **`ResNet(nn.Module)`** - `init_conv`: `Conv2d(3, 32, kernel_size=7, stride=2, padding=3)` + `BatchNorm2d(32)` + `ReLU`; then `ResidualBlock(32, 32)` (`block1`) and `ResidualBlock(32, 64)` (`block2`); `AdaptiveAvgPool2d((1, 1))`; `Linear(64, num_classes)`; output is `sigmoid`.
- **`InceptionBlock(nn.Module)`** - four parallel branches on the same input: `1x1` conv (`f1`), `3x3` conv (`f2`), `5x5` conv (`f3`), and `MaxPool2d(3, stride=1, padding=1)` followed by a `1x1` conv (`f1`). Each branch is ReLU-activated and the four outputs are concatenated along the channel dimension.
- **`GoogLeNet(nn.Module)`** - `Conv2d(3, 32, 3x3, padding=1)` + ReLU, one `InceptionBlock(32, 16, 16, 16)` (so 64 channels out), `AdaptiveAvgPool2d((1, 1))`, `Linear(64, num_classes)`, `sigmoid`.
- **`CNN2D(nn.Module)`** - configurable baseline CNN. Constructor accepts `use_bn=True`, `dropout=0.5`, `init_type="he"`; it defines `conv1 = Conv2d(3, 32, 3x3, padding=1)` with `BatchNorm2d(32)` when `use_bn` else `nn.Identity()`, and `conv2 = Conv2d(32, 64, 3x3, padding=1)` with the matching batch-norm-or-identity. This is the model used for the ablation experiments in `train.py`; the remainder of its definition lies past the point reached while reading the file, so its later layers are not described here.

### `models_temporal.py` - recurrent models and activations

- **Activation helpers** - `step_activation(x)` returns `(x >= 0.0).float()`; `sigmoid_activation(x)` returns `torch.sigmoid(x)`; `tanh_activation(x)` returns `torch.tanh(x)`.
- **`TemporalRNN(nn.Module)`** - `nn.RNN(input_dim=4, hidden_dim=32, batch_first=True, nonlinearity="tanh")`, then `Linear(hidden_dim, num_classes)`. `forward` takes the final hidden state `h_n`, squeezes the layer dimension, applies the fully connected layer and a `sigmoid`.
- **`TemporalLSTM(nn.Module)`** - `nn.LSTM(input_dim=4, hidden_dim=32, batch_first=True)`, `Dropout(0.2)`, `Linear(hidden_dim, num_classes)`. `forward` uses the final hidden state `h_n`, applies dropout, the fully connected layer, and a `sigmoid`. Its default input dimension (`4`) matches `SequenceDataset`'s `feature_dim`.

### `models_vae.py` - variational autoencoder

The file exists and is imported by `train.py` as `from models_vae import VAE, vae_loss_fn`, so it exports a `VAE` module and a `vae_loss_fn` loss function. Its internals were not inspected in this review and are therefore not described here.

### `paradigms.py` - standalone learning rules

All functions operate on plain tensors and are independent of the `nn.Module` models above.

- **`hebbian_update(weights, x, y, lr=0.01)`** - returns `weights + lr * outer(y, x)`.
- **`ojas_update(weights, x, y, lr=0.01)`** - Oja's normalised Hebbian rule: decay term `(y ** 2).unsqueeze(1) * weights`, result `weights + lr * (outer(y, x) - decay)`.
- **`competitive_learning(weights, x, lr=0.1)`** - computes the Euclidean distance `norm(weights - x, dim=1)` to every weight row, picks the `argmin` winner, moves the winner toward `x` by `lr`, and returns `(weights, winner_idx)`.
- **`error_correction_update(weights, bias, x, y_true, lr=0.05)`** - perceptron-style rule: predicts `1.0` when `dot(weights, x) + bias >= 0` else `0.0`, computes `error = y_true - y_pred`, updates `weights += lr * error * x` and `bias += lr * error`, returning `(weights, bias, error)`.
- **`energy_function(x, weights, bias)`** - Hopfield-style energy `-0.5 * x^T W x - b . x`.
- **`energy_based_update(weights, bias, pos_data, neg_data, lr=0.01)`** - contrastive update: `weights += lr * (outer(pos, pos) - outer(neg, neg))` and `bias += lr * (pos - neg)`.

### `train.py` - optimizer factory and training loops

- **`get_optimizer(model, opt_name="adam", lr=0.001, l2=0.0)`** - selects `SGD` for `"sgd"`, `SGD(momentum=0.9)` for `"momentum"`, `SGD(momentum=0.9, nesterov=True)` for `"nesterov"`, and `Adam` otherwise. `l2` is passed as `weight_decay` in every branch.
- **`train_ablation_cnn(dataset_type="both", opt_name="adam", use_bn=True, dropout=0.5, l2=0.001, init_type="he", augment=True, epochs=3)`** - picks CUDA when available, builds loaders with `get_loaders`, instantiates `CNN2D` with the ablation flags, builds an optimizer with `lr=0.001`, and uses `nn.BCELoss`. Per batch: move images/labels to the device (labels reshaped with `unsqueeze(1)`), `zero_grad`, forward, loss, `backward`, `step`; accumulates loss and computes accuracy by thresholding outputs at `0.5`. It prints loss and accuracy per epoch and returns the trained model.
- **`train_temporal(epochs=5)`** - builds `get_sequence_loader()`, a `TemporalLSTM`, `Adam(lr=0.001)` and `BCELoss`; per epoch iterates the sequence batches with the standard forward/loss/backward/step sequence and prints the mean loss; returns the model.
- **`train_vae(dataset_type="both", epochs=3)`** - builds loaders with `augment=False`, a `VAE`, and `Adam(lr=0.001)`, then iterates epochs calling `model.train()`. The remainder of the VAE loop lies past the point reached while reading the file; the VAE loss used elsewhere in the module is `vae_loss_fn`.

### `evaluate.py` - evaluation, benchmarking and drowsiness rule

- **`evaluate_model(model, test_loader, device="cpu")`** - sets `model.eval()`, runs the loader under `torch.no_grad()`, thresholds outputs at `0.5` and flattens them, collects true labels, then computes `accuracy_score`, `precision_score`, `recall_score`, `f1_score` (all with `zero_division=0`) and `confusion_matrix`. Returns a dict with keys `accuracy`, `precision`, `recall`, `f1`, `confusion_matrix`.
- **`benchmark_fps(model, input_shape=(1, 3, 64, 64), num_runs=50, device="cpu")`** - runs 5 warm-up forward passes on a random dummy input, times `num_runs` forward passes, and returns `{"fps", "latency_ms"}` where `latency_ms = elapsed / num_runs * 1000`.
- **`compute_perclos(eye_states, threshold=0.8)`** - counts frames whose state equals `0` as closed, divides by the number of frames to get PERCLOS, and flags drowsiness when `perclos >= threshold`. Returns `(perclos, is_drowsy)`.
- **`__main__` block** - instantiates `ResNet()` and `GoogLeNet()` and prints `benchmark_fps` for each on CPU with the default `(1, 3, 64, 64)` input.

## 5. Cross-cutting design notes

- Every supervised model (`ResNet`, `GoogLeNet`, `CNN2D`, `TemporalRNN`, `TemporalLSTM`) ends in a `sigmoid` with `num_classes=1` and is trained with `BCELoss`, so this is a binary (drowsy / not drowsy) setup.
- Image models expect 3-channel inputs; the default size used across loaders, transforms and the benchmark is `64x64`.
- Temporal models expect `feature_dim=4` sequences, which is exactly what `SequenceDataset` produces.
- `paradigms.py` is a self-contained library of classical update rules; nothing in `train.py` imports it, so it serves as reference/experimental code rather than part of the trained pipeline.
- `evaluate.py` evaluates any model through the common `(model, test_loader)` interface; its `__main__` path only benchmarks `ResNet` and `GoogLeNet`.
