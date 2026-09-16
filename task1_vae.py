"""
Click Run to train using the extracted keras_png_slices_data folder beside this file.
    python task1_vae.py --epochs 50
    python task1_vae.py --mode evaluate --checkpoint PATH_TO_BEST_PT

The default 2D latent space allows direct manifold sampling. Larger latent
spaces also work: the manifold grid varies its first two coordinates while
holding others at zero, and the held-out embedding is projected using PCA.
No pretrained models. This is a standalone file; no local helper modules are needed. Source: https://arxiv.org/abs/1312.6114
ChatGPT assisted with implementation. Use --data-dir PATH if the extracted PNG dataset is elsewhere.

Defaults: 50 epochs, 128x128 images, all provided PNG slices, 2 latent dimensions.
The supplied train/validate/test folders are preserved and case overlap is rejected.
Outputs go into part4_results.
"""


import argparse
import csv
import json
import math
import random
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

# Resolve data/output paths relative to this script, regardless of the terminal folder.
ROOT = Path(__file__).resolve().parent
# PNG intensities encode categories, not numerical amounts: 0/85/170/255 -> 0/1/2/3.
MASK_VALUES = [0, 85, 170, 255]


# Accept either the dataset folder itself or the common nested extraction layouts.
# DATA LOCATION: accept the supplied extraction layouts; no images are loaded here.
def find_data_folder(root):
    root = Path(root)
    for candidate in (root, root / "keras_png_slices_data",
                      root / "keras_png_slices_data" / "keras_png_slices_data"):
        if all((candidate / f"keras_png_slices_{name}").is_dir()
               for name in ("train", "validate", "test")):
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot find the keras_png_slices train/validate/test folders under {root}. "
                            "Keep the extracted dataset beside this script or use --data-dir PATH.")


# ONE SAMPLE: read a grayscale slice on demand. The DataLoader stacks samples into batches.
# Images are floating-point intensities; segmentation masks are integer category IDs.
class PngSlices(Dataset):
    """One preprocessed MRI PNG and its paired categorical mask per sample."""
    def __init__(self, rows, image_size, class_values=None, slices_per_volume=0):
        self.rows = rows
        self.image_size = image_size
        self.class_values = class_values
        # Optional speed/coverage trade-off: take evenly spaced slices within each case.
        if slices_per_volume:
            selected = []
            for subject in sorted({row["subject_id"] for row in rows}):
                group = [row for row in rows if row["subject_id"] == subject]
                indices = np.unique(np.linspace(0, len(group)-1, min(len(group), slices_per_volume)).astype(int))
                selected.extend(group[i] for i in indices)
            self.rows = selected

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["image"]) as image:
            if image.mode != "L":
                raise ValueError(f"Expected an 8-bit grayscale MRI: {row['image']}")
            original_size = image.size
            if image.size != (self.image_size, self.image_size):
                image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            # MRI intensities become float32 in [0, 1]; add a channel axis below.
            # One sample is [1, H, W]; DataLoader stacks samples into [B, 1, H, W].
            pixels = np.asarray(image, dtype=np.float32).copy() / 255.0
        sample = dict(image=torch.from_numpy(pixels).unsqueeze(0),
                      subject_id=row["subject_id"], slice_index=row["slice_index"])
        if self.class_values is not None:
            with Image.open(row["mask"]) as mask:
                if mask.mode != "L" or mask.size != original_size:
                    raise ValueError(f"MRI/mask format or dimensions differ: {row['mask']}")
                # Nearest-neighbour resizing preserves class IDs; bilinear would invent labels.
                if mask.size != (self.image_size, self.image_size):
                    mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
                raw = np.asarray(mask).copy()
            # Convert grayscale codes to integer class indices; -1 detects unknown codes.
            mapped = np.full(raw.shape, -1, dtype=np.int64)
            for class_index, raw_value in enumerate(self.class_values):
                mapped[raw == raw_value] = class_index
            if (mapped < 0).any():
                raise ValueError(f"Unexpected mask values {np.unique(raw)} in {row['mask']}; "
                                 f"expected {self.class_values}.")
            sample["mask"] = torch.from_numpy(mapped)
        return sample


# Command-line options override these defaults without editing the training code.
# USER SETTINGS: define command-line defaults so the same file works from Run or a terminal.
def parser_for(description, epochs, batch_size, lr, slices=0):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, help="New or empty results directory")
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--slices-per-volume", type=int, default=slices,
                        help="0 uses every PNG; positive selects evenly spaced slices per case")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser


# RUN SETUP: validate settings, choose hardware/output paths and read checkpoint metadata.
# This prepares an experiment; it does not perform a training epoch.
def configure(parser, task):
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.workers < 0 or args.slices_per_volume < 0:
        parser.error("epochs/batch-size/lr must be positive; workers/slices-per-volume must be nonnegative.")
    # Powers of two support repeated halving/doubling in the encoder and decoder.
    if args.image_size < 32 or args.image_size > 256 or args.image_size & (args.image_size-1):
        parser.error("image-size must be a power of two from 32 to 256.")
    if args.mode == "evaluate" and args.checkpoint is None:
        parser.error("Evaluation requires --checkpoint PATH.")
    if args.mode == "train" and args.checkpoint is not None:
        parser.error("Use --mode evaluate to load a checkpoint; training starts a new model.")
    chosen = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = torch.device(chosen if args.device == "auto" else args.device)
    if args.device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this Python environment.")
    # Automatic mixed precision (AMP) is enabled only on CUDA in these scripts.
    args.amp = args.amp and args.device.type == "cuda"
    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint["task"] != task:
            parser.error(f"Checkpoint belongs to {checkpoint['task']}, not {task}.")
        # Restore settings needed to rebuild the saved architecture and data selection.
        for key in ("image_size", "seed", "slices_per_volume", "latent_dim", "beta", "base_channels"):
            if key in checkpoint["config"]:
                setattr(args, key, checkpoint["config"][key])
    # Seed all three random-number sources. CUDA kernels can still vary across runs.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        # Let cuDNN benchmark convolution algorithms for the chosen image dimensions.
        torch.backends.cudnn.benchmark = True
    args.data_dir = find_data_folder(args.data_dir)
    args.output = args.output or ROOT / "part4_results" / f"{task}_{args.mode}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output folder must be new or empty, to preserve earlier runs.")
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Task: {task}; device: {args.device}; mixed precision: {args.amp}", flush=True)
    if args.device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(args.device), flush=True)
    save_json(args.output / "config.json", configuration(args))
    return args, checkpoint


# Convert Path/device objects into JSON-compatible values and record the environment.
# RECORD KEEPING: convert runtime settings to values that can be written to JSON.
def configuration(args):
    config = {key: str(value) if isinstance(value, (Path, torch.device)) else value
              for key, value in vars(args).items()}
    config["torch_version"] = str(torch.__version__)
    config["gpu"] = torch.cuda.get_device_name(args.device) if args.device.type == "cuda" else None
    return config


# Use the supplied splits; training changes weights, validation selects a model,
# and held-out test cases measure its final performance. The GAN uses no validation selection.
# DATA WORKFLOW: discover files -> identify subjects -> reject split overlap -> form batches.
# A subject is a person/case; a sample is one slice. Keep these units distinct in the demo.
def make_loaders(args, segmentation=False, checkpoint=None):
    root = find_data_folder(args.data_dir)
    datasets, split = {}, {}
    # Only segmentation needs masks. The VAE and GAN learn from MRI images alone.
    class_values = MASK_VALUES.copy() if segmentation else None
    if checkpoint and segmentation and checkpoint["class_values"] != class_values:
        raise ValueError("The checkpoint's class mapping differs from this PNG dataset.")
    for name, folder_name in (("train", "train"), ("val", "validate"), ("test", "test")):
        rows = []
        for path in (root / f"keras_png_slices_{folder_name}").glob("*.png"):
            # Parse case and slice IDs so images can be paired and grouped by subject.
            match = re.fullmatch(r"case_(\d+)_slice_(\d+)\.nii\.png", path.name)
            if match is None:
                raise ValueError(f"Cannot extract case and slice IDs from {path.name}")
            case, slice_number = match.groups()
            mask = root / f"keras_png_slices_seg_{folder_name}" / f"seg_{case}_slice_{slice_number}.nii.png"
            if segmentation and not mask.is_file():
                raise FileNotFoundError(f"Paired mask is missing: {mask}")
            rows.append(dict(image=str(path), mask=str(mask),
                             subject_id=f"case_{int(case):03}", slice_index=int(slice_number)))
        rows.sort(key=lambda row: (row["subject_id"], row["slice_index"]))
        if not rows:
            raise ValueError(f"No PNG slices in {folder_name}.")
        split[name] = sorted({row["subject_id"] for row in rows})
        datasets[name] = PngSlices(rows, args.image_size, class_values, args.slices_per_volume)
    # Adjacent slices from one person are similar: sharing a case across splits leaks data.
    cases = [case for subjects in split.values() for case in subjects]
    if len(cases) != len(set(cases)):
        raise ValueError("A case ID appears in multiple splits; this would cause data leakage.")
    if checkpoint and split != checkpoint["split"]:
        raise ValueError("Current case splits differ from the checkpoint's saved split.")
    save_json(args.output / "subject_splits.json", split)
    # Shuffle only training. PNG decoding/resizing happens on CPU, then batches move
    # to the selected device in the training loop; this is not a GPU dataset cache.
    # Pinned CPU memory helps CUDA transfers; persistent workers avoid worker restarts.
    loaders = {name: DataLoader(dataset, batch_size=args.batch_size, shuffle=name == "train",
                               num_workers=args.workers, pin_memory=args.device.type == "cuda",
                               persistent_workers=args.workers > 0,
                               generator=torch.Generator().manual_seed(args.seed+i))
               for i, (name, dataset) in enumerate(datasets.items())}
    for name, dataset in datasets.items():
        print(f"{name}: {len(split[name])} cases, {len(dataset)} slices", flush=True)
    return loaders, split, class_values


# GPU launches are asynchronous; synchronise before reading a wall-clock timer.
# TIMING: CPU code can continue while CUDA runs; wait before taking a meaningful timestamp.
def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# REPORT OUTPUT: write readable measurements/settings; this does not alter model weights.
def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


# Save weights plus the settings/splits needed for evaluation. Optimiser state is
# not included, so these checkpoints are not complete training-resume snapshots.
# CHECKPOINT OUTPUT: save learned weights and the metadata needed to interpret them.
# A state_dict contains parameters and registered buffers, not the Python model class itself.
def save_model(path, task, args, split, epoch, **payload):
    torch.save(dict(task=task, config=configuration(args), split=split, epoch=epoch, **payload), path)


# CSV preserves exact epoch values; the PNG makes the learning trends easy to inspect.
# LEARNING CURVES: record epoch measurements to inspect progress and possible overfitting.
def save_history(output, history, columns):
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    fig, ax = plt.subplots(figsize=(8, 4))
    for column in columns:
        ax.plot([row["epoch"] for row in history], [row[column] for row in history], label=column)
    ax.set(xlabel="Epoch", ylabel="Loss / score")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=150)
    plt.close(fig)


# Detach removes autograd tracking; matplotlib needs CPU images in the display range.
# VISUAL OUTPUT: arrange images for inspection; a small grid is not a complete test evaluation.
def image_grid(images, path, columns=8, title=None):
    images = images.detach().float().cpu().clamp(0, 1)
    rows = math.ceil(len(images) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns*1.5, rows*1.5), squeeze=False)
    for index, ax in enumerate(axes.flat):
        ax.axis("off")
        if index < len(images):
            ax.imshow(images[index, 0], cmap="gray", vmin=0, vmax=1)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# Fail clearly on NaN/infinite losses instead of silently saving invalid results.
# ERROR CHECK: NaN/Inf signals an invalid numeric result; stop rather than report it as success.
def require_finite(value):
    if not math.isfinite(value):
        raise RuntimeError("Non-finite loss. Check input data or try a lower learning rate / --no-amp.")


# Model and task-specific training.
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch import nn



# VAE flow: MRI -> encoder -> Gaussian latent distribution -> sample z -> decoder.
# B = batch size; latent_dim = number of coordinates used to represent each image.
# VAE ARCHITECTURE: image -> encoder -> mean/log variance -> sampled z -> decoder -> image.
# For defaults: [B,1,128,128] -> [B,256,8,8] -> two [B,2] heads -> reconstruction.
# There is no mask target: the input MRI itself supplies the reconstruction target.
class VAE(nn.Module):
    def __init__(self, image_size=128, latent_dim=2):
        super().__init__()
        self.latent_dim, self.side = latent_dim, image_size // 16
        # Four stride-2 convolutions reduce 128x128 -> 64 -> 32 -> 16 -> 8.
        # Channels increase 1 -> 32 -> 64 -> 128 -> 256 as spatial size decreases.
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(), nn.Flatten(),
        )
        count = 256 * self.side * self.side
        # Two heads describe q(z|x): a mean and log(variance) for each latent coordinate.
        # These are learned distribution parameters, not class predictions.
        self.mean = nn.Linear(count, latent_dim)
        self.log_variance = nn.Linear(count, latent_dim)
        # Expand [B, latent_dim] back into a spatial feature map for the decoder.
        self.expand = nn.Linear(latent_dim, count)
        # Transposed convolutions double spatial size four times.
        # Sigmoid makes the reconstructed intensities lie in [0, 1], like the inputs.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid(),
        )

    def encode(self, images):
        hidden = self.encoder(images)
        # Bound log-variance to avoid extreme exponentials during sampling and KL loss.
        return self.mean(hidden), self.log_variance(hidden).clamp(-20, 20)

    def decode(self, latent):
        return self.decoder(self.expand(latent).reshape(-1, 256, self.side, self.side))

    def forward(self, images, sample=True):
        mean, log_variance = self.encode(images)
        # Reparameterisation: z = mu + sigma * epsilon, epsilon ~ N(0, I).
        # exp(0.5 * log_variance) is sigma; gradients can flow into mu and sigma.
        # Evaluation uses mu directly (sample=False), so reconstruction is deterministic.
        # exp(log variance) is variance; exp(0.5 * log variance) is standard deviation.
        # Reparameterisation keeps the noise independent while gradients reach mean and log variance.
        latent = mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean) if sample else mean
        return self.decode(latent), mean, log_variance


# OBJECTIVE: accurate pixels (MSE) plus a latent prior penalty (KL).
# KL is summed over latent coordinates, averaged over images, then scaled by pixel count here.
# A lower total loss is a trade-off between reconstruction and regularisation, not an accuracy.
def vae_loss(reconstructed, images, mean, log_variance, beta):
    # Float32 loss arithmetic keeps the KL exponential stable under autocast.
    # MSE averages squared reconstruction error over every pixel and every image.
    reconstruction = F.mse_loss(reconstructed.float(), images.float())
    mean, log_variance = mean.float(), log_variance.float()
    # Closed-form KL(q(z|x) || N(0,I)): sum latent coordinates, then average images.
    # It encourages a latent distribution that can be sampled smoothly from the prior.
    # If mean=0 and log_variance=0, the posterior equals the unit Gaussian and KL is zero.
    kl = -0.5 * (1 + log_variance - mean.square() - log_variance.exp()).sum(1).mean()
    # Both summed image error and KL are divided by the number of image pixels.
    # beta controls the reconstruction/regularisation trade-off; logged KL is unscaled.
    objective = reconstruction + beta * kl / images[0].numel()
    return objective, reconstruction, kl


# EPOCH WORKFLOW: MRI batch -> reconstruct -> compute MSE/KL -> update only during training.
# The sample flag uses noise during training and the posterior mean during evaluation.
def epoch(model, loader, args, beta, optimizer=None, scaler=None):
    # Supplying an optimiser enables training; without one this is evaluation only.
    training = optimizer is not None
    model.train(training)
    totals = np.zeros(3)
    count = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            images = batch["image"].to(args.device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            # Autocast chooses precision per operation; sensitive loss arithmetic stays float32.
            with torch.autocast(device_type=args.device.type, enabled=args.amp):
                reconstructed, mean, log_variance = model(images, sample=training)
                loss, reconstruction, kl = vae_loss(reconstructed, images, mean, log_variance, beta)
            if training:
                # Scale before backprop to protect small FP16 gradients, then update weights.
                # GradScaler checks for overflow and adapts its scale for subsequent batches.
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            # Weight batch averages by batch size so the smaller final batch counts fairly.
            # .item() copies scalar metrics to CPU and synchronises CUDA here.
            totals += np.array([loss.item(), reconstruction.item(), kl.item()]) * len(images)
            count += len(images)
    values = (totals / count).tolist()
    for value in values:
        require_finite(value)
    return dict(loss=values[0], mse=values[1], kl=values[2])


# THREE VIEWS: paired reconstructions, a decoded latent grid, and encoded test-image positions.
# The latent grid starts from chosen coordinates; the embedding starts from real images.
@torch.inference_mode()
def visualise(model, loader, args):
    model.eval()
    batch = next(iter(loader))
    originals = batch["image"][:8].to(args.device)
    reconstructed, _, _ = model(originals, sample=False)
    # Interleave each input with its reconstruction to make comparison easier.
    paired = torch.stack((originals, reconstructed), dim=1).flatten(0, 1)
    image_grid(paired, args.output / "reconstructions.png", columns=4,
               title="Alternating original / reconstruction")
    # Decode a 12x12 grid in the first two latent coordinates to visualise the manifold.
    # For larger latent spaces, all remaining coordinates are held at zero.
    # Changing one latent coordinate at a time lets us inspect how the learned decoder varies images.
    positions = torch.linspace(-2.5, 2.5, 12, device=args.device)
    latent = torch.zeros(144, args.latent_dim, device=args.device)
    for index, (y, x) in enumerate(torch.cartesian_prod(positions.flip(0), positions)):
        latent[index, :2] = torch.stack((x, y))
    manifold = torch.cat([model.decode(z) for z in latent.split(args.batch_size)])
    image_grid(manifold, args.output / "latent_manifold.png", columns=12,
               title="Latent coordinates 1 and 2: -2.5 to +2.5; other coordinates = 0")
    means, subject_ids = [], []
    for batch in loader:
        mean, _ = model.encode(batch["image"].to(args.device))
        means.append(mean.cpu().numpy())
        subject_ids.extend(batch["subject_id"])
    means = np.concatenate(means)
    if len(means) >= 2:
        # Plot posterior means directly in 2D, or project them using PCA for display only.
        embedding = means if args.latent_dim == 2 else PCA(n_components=2).fit_transform(means)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(embedding[:, 0], embedding[:, 1], s=8, alpha=0.5)
        ax.set(xlabel="Latent 1" if args.latent_dim == 2 else "PCA 1",
               ylabel="Latent 2" if args.latent_dim == 2 else "PCA 2",
               title="Held-out MRI latent means")
        fig.tight_layout()
        fig.savefig(args.output / "latent_embedding.png", dpi=150)
        plt.close(fig)
    np.savez_compressed(args.output / "latent_means.npz", means=means,
                        subject_ids=np.asarray(subject_ids, dtype=str))


# DEMO ROUTE: build VAE -> train with KL warm-up -> select fixed-beta validation loss -> test.
# The selected epoch may be earlier than the last epoch; test does not select the checkpoint.
def main():
    parser = parser_for(__doc__, epochs=50, batch_size=32, lr=1e-3)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--kl-warmup", type=int, default=10)
    args, checkpoint = configure(parser, "vae")
    if args.latent_dim < 2 or args.beta < 0 or args.kl_warmup < 1:
        parser.error("latent-dim >= 2, beta >= 0 and kl-warmup >= 1 are required.")
    loaders, split, _ = make_loaders(args, checkpoint=checkpoint)
    model = VAE(args.image_size, args.latent_dim).to(args.device)
    training_seconds = None
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
        best, history = float("inf"), []
        sync(args.device)
        start = time.perf_counter()
        for number in range(1, args.epochs + 1):
            # KL warm-up gradually adds regularisation while the decoder learns to reconstruct.
            # With defaults, beta is 0.1 in epoch 1 and reaches 1.0 at epoch 10.
            beta = args.beta * min(1, number / args.kl_warmup)
            train = epoch(model, loaders["train"], args, beta, optimizer, scaler)
            # A fixed beta and posterior-mean reconstruction make selection consistent.
            val = epoch(model, loaders["val"], args, args.beta)
            history.append(dict(epoch=number, train_loss=train["loss"], val_loss=val["loss"],
                                train_mse=train["mse"], train_kl=train["kl"], beta=beta))
            # Select the lowest validation objective; test data never chooses the checkpoint.
            if val["loss"] < best:
                best = val["loss"]
                save_model(args.output / "best.pt", "vae", args, split, number,
                           model=model.state_dict(), validation_loss=best)
            save_history(args.output, history, ["train_loss", "val_loss"])
            print(f"Epoch {number}/{args.epochs}: train={train['loss']:.5f}, val={val['loss']:.5f}, "
                  f"MSE={train['mse']:.5f}, KL={train['kl']:.3f}", flush=True)
        sync(args.device)
        training_seconds = time.perf_counter() - start
        checkpoint = torch.load(args.output / "best.pt", map_location=args.device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
    # Evaluate the selected checkpoint, using posterior means rather than random samples.
    result = epoch(model, loaders["test"], args, args.beta)
    visualise(model, loaders["test"], args)
    result.update(selected_epoch=checkpoint["epoch"], training_seconds=training_seconds,
                  evaluation="Held-out subjects; posterior-mean reconstruction, not a sampled ELBO estimate.",
                  image_size=args.image_size, slices_per_volume=args.slices_per_volume)
    save_json(args.output / "test_results.json", result)
    print("Test results:", result)
    print("Results:", args.output.resolve())


# Run training/evaluation only when launched directly, not when imported.
# This guard also matters when DataLoader starts worker processes on Windows.
if __name__ == "__main__":
    main()
