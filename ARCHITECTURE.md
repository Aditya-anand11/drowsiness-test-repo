# Architecture and Code-Level Functionality Guide

This document explains the core technology and the functionality of each source file in this repository. It is based on a direct read of the code, not on the project README.

## 1. What this repository actually is

The repository is a **Python / PyTorch machine-learning codebase for driver-drowsiness detection**. It contains image/sequence data loading, several model families (CNN, temporal RNN/LSTM, VAE), a library of classical learning rules, a training script, and an evaluation/benchmarking script.

> **Important discrepancy:** `readme.md` describes a "full-stack TypeScript application" (Next.js, Express, Prisma, PostgreSQL, Redis, BullMQ, Zod, pino). **None of that technology exists in this repository.** There is no TypeScript, no API server, no database and no frontend here. Everything below is derived from the actual `.py` files.

## 2. Core technology

| Area | Technology present in the code |
| --- | --- |
| Deep learning | PyTorch (`torch`, `torch.nn`, `torch.nn.functional`, `torch.optim`, `torch.utils.data`) |
| Image handling | Pillow (`PIL.Image`), `torchvision.datasets.ImageFolder`, `torchvision.transforms` |
| Metrics | scikit-learn (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`) |
| Numerics | NumPy |
| Data plumbing | Python `glob`, `os`, `time` |

## 3. High-level architecture

```text
 image folders on disk ("train", "test", "no yawn", "yawn")
        |
        v
dataloader.py  -- FolderImageDataset / ImageFolder + transforms + random_split
        |            -> DataLoader batches (images, labels) and synthetic sequences
        v
models_cnn.py  -- CNN2D | ResNet (ResidualBlock) | GoogLeNet (InceptionBlock)
models_temporal.py -- TemporalRNN | TemporalLSTM
models_vae.py  -- VAE (conv encoder / transposed-conv decoder) + vae_loss_fn
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

- **`FolderImageDataset(Dataset)`** - generic two-folder binary dataset. The constructor takes `class0_dir` and `class1_dir`, globs `*.*` in each folder, and stores `(path, label)` pairs with label `0` for the first folder and label `1` for the second. `__len__` returns the number of samples. `__getitem__` opens the image with Pillow, converts it to `RGB`, applies the transform (if any) and returns `(image, label)`.
- **`get_transforms(img_size=(64, 64), augment=True)`** - returns a `(train_tf, test_tf)` pair of `torchvision.transforms.Compose` pipelines. When `augment` is true the training pipeline is `Resize -> RandomRotation(15) -> RandomHorizontalFlip -> ColorJitter(brightness=0.2) -> ToTensor`; otherwise it is just `Resize -> ToTensor`. The test pipeline is always `Resize -> ToTensor` (no augmentation).
- **`get_loaders(dataset_type="both", img_size=(64, 64), batch_size=32, augment=True)`** - builds the transforms, then loads two sources: an eye dataset via `datasets.ImageFolder` for the `"train"` and `"test"` directories, and a yawn dataset via `FolderImageDataset("no yawn", "yawn")`, which is split 80/20 into train/test with `torch.utils.data.random_split`. Depending on `dataset_type` (`"eyes"`, `"yawn"`, or anything else meaning `"both"`) it returns the corresponding train/test `DataLoader`s, concatenating both sources with `ConcatDataset` in the `"both"` case. The train loader shuffles, the test loader does not.
- **`SequenceDataset(Dataset)`** - synthetic sequence dataset for the temporal models. It generates `X = torch.randn(num_samples, seq_len, feature_dim)` (defaults 500 x 10 x 4) and derives binary labels `y` from whether the mean of the first feature over time is greater than zero. `__getitem__` returns one `(sequence, label)` pair.
- **`get_sequence_loader(batch_size=32)`** - wraps `SequenceDataset` in a shuffled `DataLoader`.

### `models_cnn.py` - convolutional image models

- **`ResidualBlock(nn.Module)`** - the residual building block used by the ResNet. It holds two 3x3 `Conv2d` layers each followed by `BatchNorm2d`, and a `shortcut`. The shortcut is an empty `nn.Sequential` when `in_c == out_c`; otherwise it is a 1x1 `Conv2d` plus `BatchNorm2d` so that channel dimensions match. `forward` computes the shortcut of the input, applies `relu(bn1(conv1(x)))`, `bn2(conv2(...))`, then adds the residual and applies `relu` (standard residual connection).
- **`ResNet(nn.Module)`** - a small ResNet. An initial `Conv2d(3, 32, kernel_size=7, stride=2, padding=3)` + `BatchNorm2d` + `ReLU`, followed by `ResidualBlock(32, 32)` and `ResidualBlock(32, 64)`, a `AdaptiveAvgPool2d((1, 1))`, and a final `Linear(64, num_classes)` (default `num_classes=1`). After the final linear layer it applies `torch.sigmoid`, so the model outputs a probability suitable for binary `BCELoss`.
- **`InceptionBlock(nn.Module)`** - an Inception-style block with four parallel branches over the same input: a 1x1 conv (`f1` filters), a 3x3 conv (`f2`), a 5x5 conv (`f3`), and a `MaxPool2d(3, stride=1, padding=1)` followed by a 1x1 conv (`f1`). All branches are passed through `F.relu` and concatenated along the channel dimension.
- **`GoogLeNet(nn.Module)`** - a compact GoogLeNet variant: `Conv2d(3, 32, kernel_size=3, padding=1)` + `ReLU`, one `InceptionBlock(32, 16, 16, 16)`, `AdaptiveAvgPool2d((1, 1))`, `flatten`, and `Linear(16 * 4, num_classes)`, followed by `torch.sigmoid`.
- **`CNN2D(nn.Module)`** - the configurable baseline CNN used by the ablation study. Its constructor parameters are `use_bn`, `dropout` and `init_type` (weight initialisation scheme, default `"he"`). It has two convolutional stages: `Conv2d(3, 32, kernel_size=3, padding=1)` followed by a batch-norm layer, then `Conv2d(32, 64, kernel_size=3, padding=1)` followed by another batch-norm layer. When `use_bn` is false the batch-norm modules are replaced with `nn.Identity()`, so the same architecture can be run with and without normalisation. Like the other CNNs, it ends in a sigmoid output for binary classification.

### `models_temporal.py` - recurrent temporal models

- **Activation helpers** - `step_activation(x)` returns a float tensor where values `>= 0` become `1.0` and the rest `0.0`; `sigmoid_activation(x)` returns `torch.sigmoid(x)`; `tanh_activation(x)` returns `torch.tanh(x)`.
- **`TemporalRNN(nn.Module)`** - wraps `nn.RNN(input_dim, hidden_dim, batch_first=True, nonlinearity="tanh")` (defaults `input_dim=4`, `hidden_dim=32`) followed by `nn.Linear(hidden_dim, num_classes)`. `forward` runs the sequence, takes the final hidden state `h_n`, squeezes the layer dimension, applies the linear layer and returns `torch.sigmoid` of the result.
- **`TemporalLSTM(nn.Module)`** - the LSTM counterpart: `nn.LSTM(input_dim, hidden_dim, batch_first=True)`, a `Dropout(0.2)`, and `nn.Linear(hidden_dim, num_classes)`. `forward` uses the final hidden state `h_n`, applies dropout, the linear layer and a sigmoid.

### `models_vae.py` - variational autoencoder

- **`VAE(nn.Module)`** - a convolutional VAE with `latent_dim=16`. The encoder is a `Sequential` of `Conv2d(3, 32, kernel_size=4, stride=2, padding=1)` + `ReLU`, `Conv2d(32, 64, kernel_size=4, stride=2, padding=1)` + `ReLU`, then `Flatten`. Two linear heads, `fc_mu` and `fc_logvar`, map the flattened features (`64 * 16 * 16`) to the latent dimension. The decoder starts from `decoder_fc` (`Linear(latent_dim, 64 * 16 * 16)`), reshapes to `(-1, 64, 16, 16)`, and applies `ConvTranspose2d(64, 32, ...)` + `ReLU` followed by `ConvTranspose2d(32, 3, ...)` + `Sigmoid` to reconstruct the image.
- **`reparameterize(mu, logvar)`** - the VAE reparameterisation trick: `std = exp(0.5 * logvar)`, `eps = randn_like(std)`, returning `mu + eps * std`.
- **`forward(x)`** - encodes `x` to `mu`/`logvar`, samples `z`, decodes it and returns `(recon_x, mu, logvar)`.
- **`generate(num_samples=5, device="cpu")`** - samples latent vectors with `torch.randn` and decodes them through the same decoder path, returning generated images.
- **`vae_loss_fn(recon_x, x, mu, logvar)`** - combines a summed binary cross-entropy reconstruction term (`F.binary_cross_entropy(..., reduction="sum")`) with the analytic KL divergence `-0.5 * sum(1 + logvar - mu^2 - exp(logvar))`, normalised by the batch size (`/ x.size(0)`).

### `train.py` - training entry points

- **`get_optimizer(model, opt_name="adam", lr=0.001, l2=0.0)`** - optimizer factory. `"sgd"` -> `optim.SGD` with weight decay `l2`; `"momentum"` -> SGD with `momentum=0.9`; `"nesterov"` -> SGD with `momentum=0.9, nesterov=True`; anything else falls back to `optim.Adam`. The L2 coefficient is passed as `weight_decay`.
- **`train_ablation_cnn(dataset_type="both", opt_name="adam", use_bn=True, dropout=0.5, l2=0.001, init_type="he", augment=True, epochs=3)`** - selects CUDA when available, builds the loaders via `get_loaders`, instantiates `CNN2D` with the ablation settings, builds the optimizer with `get_optimizer`, and trains with `nn.BCELoss`. Each batch moves images and `float().unsqueeze(1)` labels to the device, runs zero-grad / forward / loss / backward / step, accumulates the loss, and counts accuracy with a `> 0.5` decision threshold. It prints per-epoch loss and accuracy and returns the trained model.
- **`train_temporal(epochs=5)`** - trains `TemporalLSTM` on the synthetic sequence loader with `Adam(lr=0.001)` and `BCELoss`, printing the average epoch loss and returning the model.
- **`train_vae(dataset_type="both", epochs=3)`** - builds the non-augmented loaders via `get_loaders(..., augment=False)`, instantiates the `VAE`, uses `Adam(lr=0.001)`, and trains with the VAE objective imported from `models_vae` (`vae_loss_fn`, which combines the reconstruction BCE with the KL term). It returns the trained model.

### `evaluate.py` - evaluation and measurement

- **`evaluate_model(model, test_loader, device="cpu")`** - puts the model in `eval()` mode and runs it under `torch.no_grad()`. For each batch it thresholds the sigmoid output at `0.5` (`(outputs > 0.5).int()`), flattens the predictions, and collects them together with the true labels. It then computes accuracy, precision, recall, F1 (all with `zero_division=0`) and the confusion matrix with scikit-learn, returning them in a dict.
- **`benchmark_fps(model, input_shape=(1, 3, 64, 64), num_runs=50, device="cpu")`** - measures inference speed. After putting the model in `eval()` mode and running 5 warm-up passes under `no_grad`, it times `num_runs` forward passes on a random dummy input and returns `{'fps': num_runs / elapsed, 'latency_ms': elapsed / num_runs * 1000}`.
- **`compute_perclos(eye_states, threshold=0.8)`** - the classical drowsiness rule. It counts closed frames (`eye_states == 0`), divides by the number of frames to obtain the PERCLOS ratio, and flags drowsiness when `perclos >= threshold`. It returns `(perclos, is_drowsy)`.
- **`__main__` block** - instantiates `ResNet()` and `GoogLeNet()` on CPU and prints their `benchmark_fps` results, giving a quick throughput comparison between the two CNN architectures.

### `paradigms.py` - classical learning-rule library

These are standalone tensor-level weight-update functions (not `nn.Module`s), usable on plain `torch` tensors:

- **`hebbian_update(weights, x, y, lr=0.01)`** - Hebbian rule: `weights + lr * outer(y, x)`.
- **`ojas_update(weights, x, y, lr=0.01)`** - Oja's normalised Hebbian rule: it computes a decay term `(y ** 2).unsqueeze(1) * weights` and returns `weights + lr * (outer(y, x) - decay)`, which keeps the weights bounded.
- **`competitive_learning(weights, x, lr=0.1)`** - winner-take-all competitive learning. It computes the Euclidean distance from `x` to every row of `weights` with `torch.norm(..., dim=1)`, picks the winning row with `argmin`, moves that row towards `x` by `lr`, and returns the updated `weights` plus the winner index.
- **`error_correction_update(weights, bias, x, y_true, lr=0.05)`** - a perceptron-style rule. It predicts `1.0` when `dot(weights, x) + bias >= 0` else `0.0`, computes the error, and applies `weights += lr * error * x` and `bias += lr * error`, returning `(weights, bias, error)`.
- **`energy_function(x, weights, bias)`** - the Hopfield-style energy `-0.5 * x^T W x - b^T x`.
- **`energy_based_update(weights, bias, pos_data, neg_data, lr=0.01)`** - a contrastive/energy-based update using positive and negative phase outer products: `weights += lr * (outer(pos, pos) - outer(neg, neg))` and `bias += lr * (pos - neg)`.

## 5. How the pieces fit together

1. **Data layer** (`dataloader.py`) provides two kinds of inputs: real image classification data (eye and yawn folders, optionally combined) and a synthetic sequence stream for the temporal experiments.
2. **Model layer** (`models_cnn.py`, `models_temporal.py`, `models_vae.py`) provides spatial classifiers (CNN2D for ablations, ResNet and GoogLeNet for stronger baselines), temporal classifiers (RNN/LSTM) and a generative VAE for reconstruction/generation.
3. **Training layer** (`train.py`) selects an optimizer, constructs the corresponding model and data loader, and runs a supervised binary-classification loop (BCE) or the VAE reconstruction+KL loop.
4. **Evaluation layer** (`evaluate.py`) measures classification quality (accuracy/precision/recall/F1/confusion matrix), runtime performance (FPS/latency) and applies the PERCLOS eye-closure heuristic to declare drowsiness.
5. **`paradigms.py`** is an orthogonal reference library of classical learning rules that documents the biological/algorithmic ideas (Hebbian, Oja, competitive, error-correction, energy-based) without being wired into the main training pipeline.
