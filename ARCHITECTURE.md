# Architecture and Code-Level Functionality Guide

This document explains the core technology and the functionality of each source file in this repository, based on a direct read of the code.

## 1. What this repository actually is

The repository is a **Python / PyTorch machine-learning codebase centered on driver-drowsiness detection**. It contains data loading, several model families (2D/3D CNN, temporal RNN/LSTM, convolutional VAE), classical unsupervised / learning-rule helpers, a training script, and an evaluation / benchmarking script.

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
models_*.py    -- CNN2D / ResNet / GoogLeNet / CNN3D / YOLOHead
                  TemporalRNN / TemporalLSTM
                  VAE
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

- **`FolderImageDataset(Dataset)`** - generic two-folder binary dataset. The constructor takes `class0_dir` and `class1_dir`, globs `*.*` in each with `glob.glob(os.path.join(dir, "*.*"))`, and stores `(path, label)` pairs with label `0` for the first folder and `1` for the second. `__len__` returns the number of samples; `__getitem__` opens the image with Pillow, forces `RGB`, applies the transform, and returns `(image, label)`.
- **`get_transforms(img_size=(64, 64), augment=True)`** - returns a `(train_tf, test_tf)` pair. When `augment` is true the training pipeline is `Resize -> RandomRotation(15) -> RandomHorizontalFlip -> ColorJitter(brightness=0.2) -> ToTensor`; otherwise `Resize -> ToTensor`. The test pipeline is always `Resize -> ToTensor`.
- **`get_loaders(dataset_type="both", img_size=(64, 64), batch_size=32, augment=True)`** - the central data entry point. It loads eye images with `torchvision.datasets.ImageFolder("train", ...)` and `ImageFolder("test", ...)`, builds the yawn dataset with `FolderImageDataset("no yawn", "yawn", ...)`, and splits the latter 80/20 into `yawn_train` / `yawn_test` using `random_split`. `dataset_type` selects `"eyes"`, `"yawn"`, or `"both"` (default), where `"both"` concatenates the two sources with `ConcatDataset`. It returns a shuffled training `DataLoader` and an unshuffled test `DataLoader`.
- **`SequenceDataset(Dataset)`** - synthetic temporal dataset: `num_samples x seq_len x feature_dim` Gaussian tensors, with the label defined as whether the mean of the first feature channel over time is greater than zero, then shaped to `(batch, 1)`. Defaults are 500 samples, length 10, 4 features.
- **`get_sequence_loader(batch_size=32)`** - wraps `SequenceDataset` in a shuffled `DataLoader`.

### `models_cnn.py` - 2D/3D convolutional models

- **`ResidualBlock(nn.Module)`** - two `Conv2d` 3x3 layers with `BatchNorm2d`; when `in_c != out_c` the identity shortcut becomes a 1x1 convolution plus batch norm. The forward pass adds the shortcut to the second normalized convolution and applies ReLU.
- **`ResNet(nn.Module)`** - a compact residual network: 7x7 stride-2 initial convolution (3 -> 32, BN, ReLU), `ResidualBlock(32, 32)`, `ResidualBlock(32, 64)`, `AdaptiveAvgPool2d((1, 1))`, a linear layer (`nn.Linear(64, num_classes)`), and a sigmoid output.
- **`InceptionBlock(nn.Module)`** - four parallel branches (1x1 conv, 3x3 conv, 5x5 conv, and max-pool + 1x1 conv), each passed through ReLU and concatenated along the channel dimension.
- **`GoogLeNet(nn.Module)`** - a 3x3 convolution (3 -> 32), one `InceptionBlock(32, 16, 16, 16)`, adaptive average pooling, a linear layer sized to the concatenated branch channels (`nn.Linear(16 * 4, num_classes)`), and a sigmoid.
- **`CNN2D(nn.Module)`** - configurable baseline convolutional classifier used by the ablation trainer. It builds conv1 (3 -> 32) + optional BN + max-pool(2), conv2 (32 -> 64) + optional BN + max-pool(2), `AdaptiveAvgPool2d((4, 4))`, a dropout layer (or `nn.Identity` when `dropout == 0`), `Linear(64*4*4, 32)` + ReLU, and `Linear(32, 1)` + sigmoid. The constructor calls `_init_weights(init_type)`, which iterates over `self.modules()` and has an `init_type == "xavier"` branch and otherwise calls `nn.init.kaiming_normal_(m.weight, nonlinearity="relu")`, so `init_type` is one of the ablation axes.
- **`CNN3D(nn.Module)`** - a small 3D convolutional model built from `Conv3d(3, 16, kernel_size=(3, 3, 3), padding=1)` and `Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1)`, then a linear layer and a sigmoid output.
- **`YOLOHead(nn.Module)`** - a single 1x1 `Conv2d(in_c, num_anchors * (5 + num_classes), kernel_size=1)` producing anchor-based detection predictions; defaults are `in_c=64`, `num_classes=3`, `num_anchors=2`.
- **`focal_loss(preds, targets, alpha=0.25, gamma=2.0)`** - a focal-loss helper for class imbalance; the function body builds a per-sample factor from the `alpha` and `gamma` parameters and ends with `return (alpha_t * loss).mean()`.

### `models_temporal.py` - sequence models

- **`step_activation(x)` / `sigmoid_activation(x)` / `tanh_activation(x)`** - small activation helpers; `step_activation` returns 1.0 where `x >= 0` else 0.0.
- **`TemporalRNN(nn.Module)`** - `nn.RNN(input_dim=4, hidden_dim=32, batch_first=True, nonlinearity="tanh")` followed by a `Linear(hidden_dim, num_classes)` head applied to the final hidden state, with sigmoid output.
- **`TemporalLSTM(nn.Module)`** - `nn.LSTM(input_dim=4, hidden_dim=32, batch_first=True)`, dropout (0.2) on the final hidden state, a `Linear(hidden_dim, num_classes)` head, and sigmoid output. It is the model used by `train_temporal` in `train.py`.

### `models_vae.py` - convolutional variational autoencoder

- **`VAE(nn.Module)`** - convolutional VAE for 64x64 RGB inputs. The encoder is two stride-2 `Conv2d` blocks (3 -> 32 -> 64) with ReLU and `Flatten`, feeding `fc_mu` and `fc_logvar` `Linear(64*16*16, latent_dim=16)` layers. `reparameterize(mu, logvar)` samples `z = mu + eps * std` with `std = exp(0.5 * logvar)`. The decoder maps `latent_dim -> 64*16*16` (`decoder_fc`), reshapes to `(-1, 64, 16, 16)`, and applies two `ConvTranspose2d` blocks (64 -> 32 -> 3) with ReLU and a final Sigmoid. `forward` returns `(recon_x, mu, logvar)`.
- **`generate(num_samples=5, device="cpu")`** - samples standard-normal latents and decodes them, producing new images.
- **`vae_loss_fn(recon_x, x, mu, logvar)`** - sums binary cross-entropy reconstruction loss and the analytic KL-divergence term `-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())`, normalized by `x.size(0)`.

### `train.py` - training entry points

- **`get_optimizer(model, opt_name="adam", lr=0.001, l2=0.0)`** - optimizer factory used by the ablation study. `"sgd"` returns plain `optim.SGD`; `"momentum"` uses SGD with `momentum=0.9`; `"nesterov"` uses SGD with Nesterov momentum; any other value (including the default `"adam"`) returns `optim.Adam`. All variants pass `weight_decay=l2`.
- **`train_ablation_cnn(...)`** - trains `CNN2D` on the image loaders from `get_loaders`, moving batches to CUDA when available, using `nn.BCELoss` with labels reshaped to `(N, 1)`. Each epoch accumulates loss and thresholded accuracy at 0.5 and prints a per-epoch summary; it returns the trained model. Its parameters (`dataset_type`, `opt_name`, `use_bn`, `dropout`, `l2`, `init_type`, `augment`, `epochs`) are exactly the axes of the ablation study.
- **`train_temporal(epochs=5)`** - trains `TemporalLSTM` on the synthetic `get_sequence_loader` data with `optim.Adam` and `nn.BCELoss`, printing the mean loss per epoch.
- **`train_vae(dataset_type="both", epochs=3)`** - builds the image loaders (without augmentation) and a `VAE` with an Adam optimizer, then iterates over training epochs inside a `model.train()` loop. The part of the file that was read ends inside that epoch loop; the VAE loss is expected to come from the imported `vae_loss_fn`.

### `evaluate.py` - measurement and benchmarking

- **`evaluate_model(model, test_loader, device="cpu")`** - runs the model in `eval()` mode under `torch.no_grad()`, thresholds sigmoid outputs at 0.5, and returns a dict with accuracy, precision, recall, F1 and the confusion matrix computed with scikit-learn (`zero_division=0` for the degenerate case).
- **`benchmark_fps(model, input_shape=(1, 3, 64, 64), num_runs=50, device="cpu")`** - warms the model up with 5 forward passes, times `num_runs` passes with `time.time()`, and returns `{"fps", "latency_ms"}`.
- **`compute_perclos(eye_states, threshold=0.8)`** - the classic PERCLOS drowsiness rule: counts the frames whose state equals `0` (closed eyes), divides by the number of frames, and flags `is_drowsy` when the ratio is `>= threshold`.
- The `__main__` block benchmarks `ResNet` and `GoogLeNet` and prints their FPS/latency.

### `paradigms.py` - classical learning rules

Standalone tensor utilities illustrating unsupervised and error-correcting learning rules (no dataset or model dependency):

- **`hebbian_update(weights, x, y, lr=0.01)`** - `weights + lr * torch.outer(y, x)`.
- **`ojas_update(weights, x, y, lr=0.01)`** - Hebbian term with Oja's decay `(y ** 2).unsqueeze(1) * weights`, keeping weights normalized.
- **`competitive_learning(weights, x, lr=0.1)`** - finds the row of `weights` closest to `x` by Euclidean norm, moves that winning row toward `x`, and returns the weights plus the winner index.
- **`error_correction_update(weights, bias, x, y_true, lr=0.05)`** - a perceptron-style step: predicts 1.0 if `dot(weights, x) + bias >= 0`, computes the error, and applies the delta rule to weights and bias.
- **`energy_function(x, weights, bias)`** - the quadratic energy `-0.5 * x^T W x - b^T x` used by energy-based models.
- **`energy_based_update(weights, bias, pos_data, neg_data, lr=0.01)`** - a contrastive Hebbian update using the outer products of positive and negative data, returning the updated weights and bias.

## 5. How the pieces connect

1. `dataloader.get_loaders` assembles eye and yawn image tensors into train/test `DataLoader`s; `get_sequence_loader` provides synthetic sequences for the temporal models.
2. `train.py` instantiates a model (`CNN2D` for the ablation, `TemporalLSTM` for sequences, `VAE` for generation), selects an optimizer with `get_optimizer`, and optimizes BCE loss or `vae_loss_fn`.
3. `evaluate.py` consumes a trained model and loader to produce classification metrics, confusion matrices, FPS/latency numbers and the PERCLOS drowsiness decision.
4. `models_cnn.py`, `models_temporal.py` and `models_vae.py` are independent model catalogs; `paradigms.py` is a separate educational module of weight-update rules that is not wired into the training scripts.
