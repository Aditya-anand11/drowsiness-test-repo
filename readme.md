# Driver-Drowsiness Detection — PyTorch Research Codebase

> **Benchmarking / training on a real machine.** This repo ships a standalone
> benchmark harness, `benchmark.py`, plus `requirements.txt`. It is *not* wired
> into CI on purpose: `torch`/`torchvision` are heavy dependencies that cannot be
> installed in the restricted sandbox. To run it on a dedicated machine:
>
> ```bash
> pip install -r requirements.txt \
>   --extra-index-url https://download.pytorch.org/whl/cpu   # CPU-only torch
> python benchmark.py --dry-run                  # wiring check, no torch needed
> python benchmark.py --batch-size 8 --runs 50    # inference FPS / latency
> python benchmark.py --train-steps 20 --json bench.json   # + timed training steps
> ```
>
> `benchmark.py` benchmarks `CNN2D`, `ResNet` and `GoogLeNet` from
> `models_cnn.py` on a `(batch, 3, 64, 64)` input and, with `--train-steps`,
> times real `CNN2D` optimization steps (BCE loss, labels shaped `(B, 1)` as in
> `train.py`). It prints a table and can dump a JSON report. Training itself is
> still launched through `train.py`.

This repository is a **Python / PyTorch machine-learning codebase for vision-based driver-drowsiness detection**. It contains data loading, several model families (2D CNN, residual CNN, an Inception-style CNN, a 3D CNN, an RNN/LSTM over sequences, and a VAE), a library of classical learning-rule helpers, a training script with an ablation-friendly optimizer factory, and an evaluation / benchmarking script.

> **Note on the previous README:** an earlier revision of this file described a "full-stack TypeScript application" (Next.js, Express, Prisma, PostgreSQL, Redis, BullMQ, Zod, pino). **None of that exists in this repository.** There is no TypeScript, no HTTP API, no database, and no web frontend here. This document was rewritten from a direct read of the actual `.py` sources.

---

## 1. Core technology

| Area | Technology actually used in the code | Where |
| --- | --- | --- |
| Deep learning | PyTorch — `torch`, `torch.nn`, `torch.nn.functional`, `torch.optim`, `torch.utils.data` | all model files, `train.py` |
| Image I/O | Pillow (`PIL.Image`) | `dataloader.py` |
| Dataset plumbing | `torchvision.datasets.ImageFolder`, `torchvision.transforms`, `random_split`, `ConcatDataset`, `DataLoader` | `dataloader.py` |
| File discovery | `glob`, `os` | `dataloader.py` |
| Metrics | scikit-learn — `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix` | `evaluate.py` |
| Numerics / PERCLOS | NumPy | `evaluate.py` |
| Timing / FPS | `time` | `evaluate.py` |

There is **no** `requirements.txt`, **no** test suite, **no** configuration file, and **no** model-checkpoint / logging infrastructure in the repository. Training hyperparameters are Python function defaults.

---

## 2. Architecture at a glance

```text
 image folders on disk
   "train"  "test"  (eye state: open / closed)
   "no yawn"  "yawn"
        |
        v
 dataloader.py
   FolderImageDataset      -> (PIL image -> RGB -> transform), label 0/1
   ImageFolder             -> torchvision folder dataset
   get_transforms()        -> train (augmented) / test (resize+toTensor)
   get_loaders()           -> random_split(80/20) + ConcatDataset -> DataLoader
   SequenceDataset         -> synthetic (num_samples, seq_len=10, feature_dim=4)
        |
        v   tensors (B, 3, 64, 64) and labels (B,)
        |
        +--> models_cnn.py      CNN2D | ResNet | GoogLeNet | CNN3D | YOLOHead | focal_loss
        +--> models_temporal.py TemporalRNN | TemporalLSTM  (input B, T=10, F=4)
        +--> models_vae.py      VAE (encoder -> mu/logvar -> reparameterize -> decoder)
        |
        v
 train.py
   get_optimizer()            sgd | momentum | nesterov | adam (default), with weight_decay=l2
   train_ablation_cnn()       BCE loss, per-epoch loss + train accuracy
   train_temporal()           BCE loss on synthetic sequences
   train_vae()                vae_loss_fn (BCE reconstruction + KL divergence)
        |
        v
 evaluate.py
   evaluate_model()           accuracy, precision, recall, F1, confusion matrix
   benchmark_fps()            warm-up + timed loop -> FPS and latency (ms)
   compute_perclos()          fraction of closed frames -> drowsiness flag

 paradigms.py
   hebbian_update | ojas_update | competitive_learning
   error_correction_update | energy_function | energy_based_update
   (standalone tensor-level learning rules; not wired into train.py)
```

The codebase is split strictly by concern: **data (`dataloader.py`) → models (`models_*.py`) → training (`train.py`) → measurement (`evaluate.py`)**, with `paradigms.py` kept as an independent library of low-level weight-update rules that does not participate in the main pipeline.

---

## 3. Repository layout

| File | Role |
| --- | --- |
| `dataloader.py` | Datasets, transforms, train/test splitting, `DataLoader` factories, synthetic sequence dataset |
| `models_cnn.py` | Convolutional models and the focal-loss helper |
| `models_temporal.py` | Recurrent models over per-frame feature sequences |
| `models_vae.py` | Variational autoencoder for reconstruction / generation |
| `paradigms.py` | Classical unsupervised & error-correction learning rules |
| `train.py` | Optimizer factory and the three training routines |
| `evaluate.py` | Classification metrics, FPS/latency benchmark, PERCLOS drowsiness rule |
| `ARCHITECTURE.md` | Companion architecture note |
| `readme.md` | This document |

---

## 4. File-by-file implementation details

### 4.1 `dataloader.py` — data ingestion

**`FolderImageDataset(Dataset)`** — a generic two-folder binary dataset.

- `__init__(class0_dir, class1_dir, transform=None)` globs `*.*` inside each directory and appends `(path, label)` pairs, with label **0** for `class0_dir` and **1** for `class1_dir`.
- `__len__` returns the number of collected files.
- `__getitem__(idx)` opens the file with Pillow, **forces `RGB`** (three channels — required by every CNN in the repo, which all start with a 3-channel convolution), applies the transform if present, and returns `(image, label)`.

**`get_transforms(img_size=(64, 64), augment=True)`** returns a `(train_tf, test_tf)` pair.

- With `augment=True` the training pipeline is `Resize(img_size) → RandomRotation(15) → RandomHorizontalFlip() → ColorJitter(brightness=0.2) → ToTensor()`.
- With `augment=False` it is only `Resize(img_size) → ToTensor()`.
- `test_tf` is always `Resize(img_size) → ToTensor()` (no augmentation, no normalization).
- Because there is **no `Normalize`** step, inputs stay in `[0, 1]`; this is consistent with the VAE decoder, which ends in `Sigmoid`, and with `BCELoss` targets of 0/1.

**`get_loaders(dataset_type="both", img_size=(64, 64), batch_size=32, augment=True)`** is the main data entry point.

1. Builds `eye_train = ImageFolder("train", transform=train_tf)` and `eye_test = ImageFolder("test", transform=test_tf)` — the eye-state data uses torchvision's folder convention with `train/` and `test/` directories on disk.
2. Builds `yawn_full = FolderImageDataset("no yawn", "yawn", transform=train_tf)` — note the yawn data uses two flat folders.
3. Splits the yawn data 80/20 with `random_split(yawn_full, [yawn_train_len, yawn_test_len])`, where `yawn_train_len = int(0.8 * len(yawn_full))`.
4. Selects the sets by `dataset_type`: `"eyes"` → eye folders only, `"yawn"` → the random split only, anything else (`"both"`, the default) → `ConcatDataset([eye_train, yawn_train])` and `ConcatDataset([eye_test, yawn_test])`.
5. Wraps them in `DataLoader(train_set, batch_size=32, shuffle=True)` and `DataLoader(test_set, batch_size=32, shuffle=False)`.

Implementation notes worth flagging:

- The **yawn split uses `train_tf` for both halves**, so the yawn "test" set is *augmented*; and `random_split` consumes the global RNG, so the split differs on every run (no `generator=` and no seed).
- Eye data has an explicit on-disk `test/` folder while yawn data is split programmatically — the two halves of `"both"` therefore have different leakage guarantees.
- Directory arguments are **relative paths** (`"train"`, `"test"`, `"no yawn"`, `"yawn"`), so the process must be started from the project root.

**`SequenceDataset(Dataset)`** — a *synthetic* sequence dataset, not real video data.

- `X = torch.randn(num_samples=500, seq_len=10, feature_dim=4)` (pure Gaussian noise).
- `y = (X[:, :, 0].mean(dim=1) > 0).float().unsqueeze(1)` — the label is the sign of the mean of the first feature over time, i.e. an exactly-learnable threshold rule on noise, giving an output shape `(B, 1)`.
- `get_sequence_loader(batch_size=32)` wraps it in a shuffled `DataLoader`.

### 4.2 `models_cnn.py` — convolutional models

All image models take `(B, 3, H, W)` input (64×64 with the default transforms) and end in a `Sigmoid` producing a single probability per sample, matching the `BCELoss` used in `train.py` with `num_classes=1`.

**`ResidualBlock(in_c, out_c)`** — two 3×3 convolutions with `padding=1`, each followed by `BatchNorm2d`, plus a projection shortcut (`1×1` conv + `BatchNorm2d`) that is only created when `in_c != out_c`. `forward` computes `res = shortcut(x)`, then `relu(bn1(conv1(x))) → bn2(conv2(...))`, and returns `relu(out + res)`.

**`ResNet(num_classes=1)`** — a deliberately small hand-written ResNet: `init_conv` = `Conv2d(3, 32, k=7, stride=2, padding=3) → BatchNorm2d(32) → ReLU`, then `block1 = ResidualBlock(32, 32)`, `block2 = ResidualBlock(32, 64)`, `AdaptiveAvgPool2d((1,1))`, `Linear(64, num_classes)`, and `sigmoid` on the output. Downsampling happens only through the stride-2 stem and the global pooling — there are no max-pool stages.

**`InceptionBlock(in_c, f1, f2, f3)`** — four parallel branches: a 1×1 conv (`f1`), a 3×3 conv with padding 1 (`f2`), a 5×5 conv with padding 2 (`f3`), and a `MaxPool2d(k=3, stride=1, padding=1)` followed by a 1×1 conv (`f1`). Each branch is ReLU-activated and the four outputs are concatenated along the channel dimension with `torch.cat([...], dim=1)`.

**`GoogLeNet(num_classes=1)`** — a single-Inception-block network: `Conv2d(3, 32, k=3, padding=1) → ReLU → InceptionBlock(32, 16, 16, 16) → AdaptiveAvgPool2d((1,1)) → Linear(16*4, num_classes) → sigmoid`. The classifier input of `16*4 = 64` follows from the concatenation of the four branches (16+16+16+16). It is a very shallow "Inception-style" model, not the original deep GoogLeNet.

**`CNN2D(use_bn=True, dropout=0.5, init_type="he")`** — the configurable model used for all ablations. Modules: `Conv2d(3, 32, k=3, padding=1)`, `BatchNorm2d(32)` (or `nn.Identity()` when `use_bn=False`), `Conv2d(32, 64, k=3, padding=1)`, `BatchNorm2d(64)`, a `Dropout(dropout)` (or `Identity` when `dropout <= 0`), `AdaptiveAvgPool2d((4, 4))`, `Linear(64*4*4, 32)`, and `Linear(32, 1)`. It exposes three ablation switches — normalization, dropout, and weight initialization — which are set by `_init_weights(init_type)`: `nn.Conv2d`/`nn.Linear` weights get `xavier_uniform_` for the `"xavier"` path and `kaiming_normal_(nonlinearity="relu")` otherwise (`"he"`), and every bias is zero-initialized.

**`CNN3D(num_classes=1)`** — a volumetric/spatio-temporal model: `Conv3d(3, 16, k=(3,3,3), padding=1)`, `Conv3d(16, 32, k=(3,3,3), padding=1)`, `AdaptiveAvgPool3d((1,1,1))`, `Linear(32, num_classes)`. It expects a 5-D tensor `(B, C, T, H, W)` rather than a single image.

**`YOLOHead(in_c=64, num_classes=3, num_anchors=2)`** — a single detection head implemented as `Conv2d(in_c, num_anchors * (5 + num_classes), kernel_size=1)`, i.e. `2 * (5 + 3) = 16` output channels per spatial location following the YOLO convention (`5` = box `x, y, w, h` + objectness). It is a building block only; there is no matching backbone, anchor assignment, or detection loss in the repository.

**`focal_loss(preds, targets, alpha=0.25, gamma=2.0)`** — a helper implementing focal loss (down-weighting easy examples via `(1 - p_t)^gamma` scaled by `alpha`). It is defined in this module but is **not imported anywhere** — `train.py` uses plain `nn.BCELoss()`.

### 4.3 `models_temporal.py` — sequence models

Three activation helpers are defined for reference / experimentation: `step_activation` (binary threshold `(x >= 0).float()`), `sigmoid_activation`, and `tanh_activation`.

**`TemporalRNN(input_dim=4, hidden_dim=32, num_classes=1)`** — an `nn.RNN(..., batch_first=True, nonlinearity="tanh")` followed by `Linear(hidden_dim, num_classes)`. `forward` runs the sequence, keeps only the final hidden state `h_n`, squeezes the layer axis, applies the linear layer and a `sigmoid`. Input shape: `(B, T, 4)`.

**`TemporalLSTM(input_dim=4, hidden_dim=32, num_classes=1)`** — `nn.LSTM(batch_first=True)`, then `Dropout(0.2)` on the final hidden state `h_n`, then `Linear(hidden_dim, num_classes)` and `sigmoid`. Only the last hidden state is used (no attention, no per-timestep output). These feature sequences have the same 4-dimensional signature as `SequenceDataset`, i.e. the intended per-frame features are 4-D. Note that the LSTM does **not** set `input_size=...` from the constructor's `input_dim` positionally but forwards it correctly as the first positional argument.

### 4.4 `models_vae.py` — variational autoencoder

**`VAE(latent_dim=16)`**

- **Encoder:** `Conv2d(3, 32, k=4, s=2, p=1) → ReLU → Conv2d(32, 64, k=4, s=2, p=1) → ReLU → Flatten`.
- **Latent heads:** `fc_mu` and `fc_logvar`, both `Linear(64*16*16, latent_dim)`. This fixes the input resolution at **64×64** (two stride-2 convolutions turn 64→32→16, and `64 * 16 * 16 = 16384` features). Any other resolution will raise a shape error.
- **Decoder:** `decoder_fc = Linear(latent_dim, 64*16*16)`, reshaped with `.view(-1, 64, 16, 16)`, then `ConvTranspose2d(64, 32, k=4, s=2, p=1) → ReLU → ConvTranspose2d(32, 3, k=4, s=2, p=1) → Sigmoid` (64×64 RGB output in `[0, 1]`).
- **`reparameterize(mu, logvar)`** — the standard reparameterization trick: `std = exp(0.5*logvar)`, `eps ~ N(0, I)`, `z = mu + eps*std`.
- **`forward(x)`** returns the triple `(recon_x, mu, logvar)`.
- **`generate(num_samples=5, device="cpu")`** samples `z ~ N(0, I)` of width `self.fc_mu.out_features` and decodes it — the sampling path for new faces/eye images.

**`vae_loss_fn(recon_x, x, mu, logvar)`** — `F.binary_cross_entropy(recon_x, x, reduction="sum")` plus the analytic KL term `-0.5 * sum(1 + logvar - mu^2 - exp(logvar))`, divided by the batch size `x.size(0)`. The BCE is *summed over pixels* while the KL is summed over latent units, so in practice the reconstruction term dominates early in training unless it is rescaled.

### 4.5 `paradigms.py` — classical learning rules

Standalone, framework-free tensor utilities (not imported by `train.py`):

- **`hebbian_update(weights, x, y, lr=0.01)`** — `W + lr * outer(y, x)` (plain Hebbian potentiation).
- **`ojas_update(weights, x, y, lr=0.01)`** — Hebbian term minus a decay term `(y²) * W`, keeping weight norms bounded (Oja's rule).
- **`competitive_learning(weights, x, lr=0.1)`** — computes the Euclidean distance of the input to every prototype row, picks `argmin` (winner-take-all), moves that prototype toward the input with `lr * (x - w_winner)`, and returns `(weights, winner_index)`. The prototype row is updated **in place**.
- **`error_correction_update(weights, bias, x, y_true, lr=0.05)`** — a perceptron step: predict `1` if `dot(w, x) + b >= 0` else `0`, compute `error = y_true - y_pred`, and apply `w += lr*error*x`, `b += lr*error`. Returns `(weights, bias, error)`.
- **`energy_function(x, weights, bias)`** — the Hopfield-style quadratic energy `-0.5 * xᵀ W x - bᵀ x`.
- **`energy_based_update(weights, bias, pos_data, neg_data, lr=0.01)`** — contrastive Hebbian update using the outer products of positive vs. negative samples, `W += lr*(outer(pos,pos) - outer(neg,neg))` and `b += lr*(pos - neg)`.

### 4.6 `train.py` — training orchestration

**`get_optimizer(model, opt_name="adam", lr=0.001, l2=0.0)`** — an optimizer factory used by the ablations:

| `opt_name` | Optimizer |
| --- | --- |
| `"sgd"` | `SGD(lr, weight_decay=l2)` |
| `"momentum"` | `SGD(lr, momentum=0.9, weight_decay=l2)` |
| `"nesterov"` | `SGD(lr, momentum=0.9, nesterov=True, weight_decay=l2)` |
| anything else (default) | `Adam(lr, weight_decay=l2)` |

**`train_ablation_cnn(dataset_type="both", opt_name="adam", use_bn=True, dropout=0.5, l2=0.001, init_type="he", augment=True, epochs=3)`** — the main training loop.

1. Selects CUDA when available, otherwise CPU.
2. Builds loaders via `get_loaders(dataset_type, augment)` and the model `CNN2D(use_bn, dropout, init_type)`; optimizer via `get_optimizer(..., lr=0.001, l2=l2)`; loss `nn.BCELoss()`.
3. For each epoch: `model.train()`, move `images` and `labels.float().unsqueeze(1)` to the device, zero gradients, forward, compute loss, backward, step.
4. Accumulates loss and computes **train accuracy** by thresholding `outputs > 0.5` and comparing with the labels; prints `Epoch [i/epochs] Loss: … Acc: …`.
5. Returns the trained model (no checkpoint is written to disk).

The `opt_name`/`use_bn`/`dropout`/`l2`/`init_type`/`augment` arguments make this function a ready-made ablation harness — each call varies one design decision of the same architecture.

**`train_temporal(epochs=5)`** — builds the synthetic sequence loader, a `TemporalLSTM`, `Adam(lr=0.001)` and `BCELoss`; each epoch runs the standard forward/loss/backward/step and prints the mean loss. No evaluation is performed.

**`train_vae(dataset_type="both", epochs=3)`** — builds loaders with `augment=False` (reconstruction targets must be un-augmented), instantiates `VAE()` and `Adam(lr=0.001)`, then runs the reconstruction loop using `vae_loss_fn`. The VAE is trained on both classes of the same folders, i.e. it models the *distribution* of the eye/yawn patches rather than the label.

### 4.7 `evaluate.py` — measurement

**`evaluate_model(model, test_loader, device="cpu")`** — runs `model.eval()` under `torch.no_grad()`, thresholds outputs at `0.5`, flattens to 1-D, and returns a dict with `accuracy`, `precision`, `recall`, `f1` (all with `zero_division=0`) and the `confusion_matrix` from scikit-learn. Labels are taken from the loader's second element, so it works directly with the loaders from `dataloader.py`.

**`benchmark_fps(model, input_shape=(1, 3, 64, 64), num_runs=50, device="cpu")`** — runs 5 warm-up forward passes on a random dummy input, then times `num_runs` forward passes with `time.time()`, returning `{"fps": num_runs/elapsed, "latency_ms": elapsed/num_runs*1000}`.

**`compute_perclos(eye_states, threshold=0.8)`** — PERCLOS-style drowsiness rule: `closed_frames = sum(eye_states == 0)`, `perclos = closed_frames / len(eye_states)`, and `is_drowsy = perclos >= threshold`. It expects a sequence of 0/1 eye states where `0` means *closed*.

**`__main__`** — instantiates `ResNet()` and `GoogLeNet()` and prints their FPS benchmarks on CPU, e.g.:

```bash
python evaluate.py
```

*(`get_loaders` is imported in this module but is not used by the current `__main__` block.)*

---

## 5. How to use the code

Data is expected in the project root, **before** running anything:

```text
train/      # eyes, ImageFolder layout: train/open/*.png, train/closed/*.png
test/       # eyes, ImageFolder layout: test/open/*.png,  test/closed/*.png
no yawn/    # flat folder, label 0
yawn/       # flat folder, label 1
```

Training (called directly from Python, since these are plain functions):

```python
from train import train_ablation_cnn, train_temporal, train_vae
from evaluate import evaluate_model

model = train_ablation_cnn(dataset_type="both", opt_name="nesterov",
                           use_bn=True, dropout=0.5, l2=0.001,
                           init_type="he", augment=True, epochs=3)

from dataloader import get_loaders
_, test_loader = get_loaders(dataset_type="both", augment=False)
print(evaluate_model(model, test_loader))
```

Benchmarking:

```bash
python evaluate.py
```

Ablation grid (one factor at a time): `dataset_type ∈ {eyes, yawn, both}` × `opt_name ∈ {sgd, momentum, nesterov, adam}` × `use_bn ∈ {True, False}` × `dropout ∈ {0.0, 0.5}` × `l2 ∈ {0.0, 0.001}` × `init_type ∈ {he, xavier}` × `augment ∈ {True, False}`.

---

## 6. Will the model overfit?

Short answer: **it can, and with the current scripts you would not be able to tell.** Here is the grounded analysis, split by model family.

### 6.1 What the defaults already do to fight overfitting

`CNN2D` — the model used by the ablation harness — is the most regularized model in the repo:

- **Dropout 0.5** before the classifier (`nn.Dropout(dropout)`, default `0.5`).
- **BatchNorm** on both conv blocks (default `use_bn=True`).
- **L2 weight decay** `0.001` (`get_optimizer(..., weight_decay=l2)`).
- **He/Kaiming initialization** (`init_type="he"`), with a Xavier alternative.
- **Data augmentation** in `get_transforms` — `RandomRotation(15)`, `RandomHorizontalFlip`, `ColorJitter(brightness=0.2)`.
- **Aggressive downsampling of the feature map** — `AdaptiveAvgPool2d((4, 4))` then `Linear(64*4*4 → 32) → Linear(32 → 1)`, so the trainable head is tiny (≈ 33 K weights in `fc1`) and the whole model is small.
- **Only 3 epochs** by default.

That combination — small model, small head, 3 epochs, dropout 0.5, augmentation, weight decay — makes catastrophic overfitting **unlikely**; if anything the more common failure mode here is **underfitting** (a 32-unit head after a 4×4 global-avg pool is a very narrow bottleneck).

### 6.2 Where overfitting is still likely

1. **The generalization gap is never measured during training.** `train_ablation_cnn` evaluates *only* the training set (`acc = correct / total` inside the training loop) and never touches `test_loader` — the loader it receives but only unpacks and discards (`train_loader, test_loader = get_loaders(...)`; `test_loader` is never used). There is no validation split, no per-epoch test metric, no early stopping, and no saved best checkpoint. So even if the model overfits, nothing in the run reports it.
2. **Small datasets + heavy augmentation = augmented-only validation.** For the yawn half of `"both"`, `get_loaders` applies `train_tf` to `yawn_full` *before* the 80/20 `random_split`, so the "test" yawn images are randomly rotated/flipped/brightness-jittered too. Reported test accuracy on those samples is neither clean nor comparable across runs.
3. **Non-repeatable splits.** `random_split` uses the global RNG with no `generator=` and there is no `torch.manual_seed(...)` anywhere, so the yawn train/test split changes per run — metrics are not reproducible, and a lucky split can look like a good model.
4. **Potential identity leakage between the eye halves.** The eye data relies on `train/` and `test/` being genuinely disjoint on disk (per-driver/per-session). The code does not verify this; if frames from the same video end up on both sides, the reported accuracy will be optimistic regardless of architecture.
5. **Class imbalance is unhandled.** `FolderImageDataset` and `ImageFolder` count whatever files exist; there is no `WeightedRandomSampler`, no `pos_weight`, and `BCELoss` is unweighted. With `nofault`-heavy folders, accuracy is dominated by the majority class — `precision`/`recall`/`F1` in `evaluate_model` exist precisely to expose this, but nothing in training optimizes for it.

### 6.3 Per-model verdict

| Model | Overfitting risk | Reasoning from the code |
| --- | --- | --- |
| `CNN2D` | **Low to moderate** | Dropout 0.5 + BN + weight decay 0.001 + augmentation + tiny `fc1`/`fc2` head + 3 epochs. Risk rises if `dropout=0.0`, `augment=False` and `l2=0.0` are chosen in the ablation grid — that configuration is effectively unregularized and *will* overfit on a small set. |
| `ResNet` | **Low (capacity-limited)** | Only two residual blocks, 32/64 channels, a single global-average-pool, and a `Linear(64 → 1)` head. Very few parameters and **no dropout**; the realistic failure mode is underfitting, not overfitting. |
| `GoogLeNet` | **Low (capacity-limited)** | One Inception block, no batch-norm, no dropout, and a `Linear(64 → 1)` classifier fed by global average pooling. Again, underfitting is the more likely failure. |
| `TemporalRNN` / `TemporalLSTM` | **Meaningless on the current data** | Trained on `SequenceDataset`, which is i.i.d. Gaussian noise with a threshold label. There is no real temporal signal to memorize beyond the `mean(feature 0) > 0` rule, so the model converges to a near-trivial decision boundary; the LSTM's `Dropout(0.2)` and 32-unit hidden state are far more than that rule needs. Any reported loss is not evidence about driving behaviour. |
| `CNN3D` | **Untested** | Three conv layers/16→32 channels and a `Linear(32 → 1)` head. There is no 3-D dataloader in the repo, so it cannot currently be trained at all — overfitting is moot until video tensors exist. |
| `VAE` | **Moderate to high (memorization)** | A 16-D latent for 64×64 RGB patches is a strong bottleneck, which limits memorization; however the loss is `sum(BCE)/batch + KL`, which is dominated by the reconstruction term, so the model can under-regularize its KL and behave close to a plain autoencoder (memorizing the training frames). Nothing monitors the ELBO on held-out images. |
| `paradigms.py` rules | **N/A** | Pure tensor update rules; they are never trained by the pipeline. |

### 6.4 Concrete recommendations

1. Add a **validation split** and log train **and** validation loss/F1 every epoch; the code already returns the model, so `evaluate_model` can be called on a validation loader between epochs.
2. Add `torch.manual_seed(...)` and pass `generator=torch.Generator().manual_seed(seed)` to `random_split` for reproducible, comparable ablations.
3. Use `test_tf` for the yawn test half — split the **dataset paths first**, then apply the train transform to the train subset only.
4. Track the **gap** `train_acc − test_f1` across the ablation grid; a widening gap is the overfitting signal the current instrumentation is missing.
5. Consider class weighting (`pos_weight`) or a weighted sampler if the eye/yawn folders are imbalanced, and report precision/recall/F1 rather than accuracy alone.
6. For the VAE, weight the KL term (e.g. β-VAE) or normalize BCE per pixel, and monitor reconstruction error on unseen frames.

---

## 7. Known limitations and caveats

- **No dependencies manifest.** Install manually: `torch`, `torchvision`, `Pillow`, `numpy`, `scikit-learn`.
- **No tests, no CI, no seeding, no checkpointing, no logging framework.** All hyperparameters are function defaults.
- **`train.py` exposes functions only** — there is no top-level `argparse`/CLI, so experiments are driven from a Python REPL or a driver script.
- **`focal_loss` in `models_cnn.py` is dead code** (never imported); every training loop uses `nn.BCELoss()`.
- **`YOLOHead` has no backbone, anchors, or detection loss**, and `CNN3D` has no 3-D data loader — both are building blocks for future work.
- **`benchmark_fps` does not call `torch.cuda.synchronize()`**, so on GPU the reported FPS/latency is measured asynchronously and can be optimistic.
- **`compute_perclos` uses a default threshold of `0.8`** (80 % of frames closed) and counts `eye_states == 0` as closed; both the convention and the threshold must match the eye-classifier's label convention (`ImageFolder` sorts folders alphabetically, so "closed" vs "open" ordering determines which label means closed).
- **Input-resolution coupling:** the VAE hard-codes 64×64; the CNNs adapt via pooling but were designed around `img_size=(64, 64)`.
- **`evaluate.py` imports `get_loaders` but `__main__` only benchmarks models.**
- **No normalization layer** in the transforms — inputs stay in `[0, 1]`, which the sigmoid-output models and the VAE reconstruction loss assume.
