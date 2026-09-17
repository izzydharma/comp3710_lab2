"""Part 4 Task 3: train a WGAN-GP to generate OASIS brain MRI slices.

Click Run to train using keras_png_slices_data beside this file.
    python task3_gan.py --epochs 100
    python task3_gan.py --mode evaluate --checkpoint PATH_TO_LAST_PT

Saves fixed-noise samples throughout training, both loss curves, a generator/
critic checkpoint, and diversity/nearest-training-preview diagnostics.
These diagnostics do not prove realism or rule out mode collapse. The lab
requires convincing OASIS images and explanation during the demonstration.
No pretrained models. This is a standalone file; no local helper modules are needed. Sources: https://arxiv.org/abs/1704.00028 and https://arxiv.org/abs/1511.06434
ChatGPT assisted with implementation. Use --data-dir PATH if the extracted PNG dataset is elsewhere.

Defaults: 100 epochs, 128x128 images, batch 32, Adam lr=1e-4, betas=(0,0.9),
5 critic batches per generator update and gradient-penalty weight 10.
One epoch visits the training images once; the final partial critic group also
gets a generator update. Full float32 is the default for stable double backward.
--amp optionally accelerates G; the critic and gradient penalty stay float32.
WGAN-GP requires NEW training; old BCE/spectral-normalised checkpoints are rejected.
It can improve training stability, but does not guarantee removal of mode collapse.
The supplied validation/test folders stay held out. Results and model weights
go into part4_results; no additional source files are required.
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
        if checkpoint.get("algorithm") != "wgan-gp":
            parser.error("This is not a WGAN-GP checkpoint. Start a new training run; old GAN weights are incompatible.")
        # Restore settings needed to rebuild the saved architecture and data selection.
        for key in ("image_size", "seed", "slices_per_volume", "latent_dim", "beta", "base_channels", "n_critic", "gp_weight"):
            if key in checkpoint["config"]:
                setattr(args, key, checkpoint["config"][key])
    if args.n_critic < 1 or not math.isfinite(args.gp_weight) or args.gp_weight <= 0:
        parser.error("n-critic must be positive and gp-weight must be finite and positive.")
    # Seed all three random-number sources. CUDA kernels can still vary across runs.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        # Let cuDNN benchmark convolution algorithms for the chosen image dimensions.
        torch.backends.cudnn.benchmark = True
    args.data_dir = find_data_folder(args.data_dir)
    args.output = args.output or ROOT / "part4_results" / f"{task}_wgan_gp_{args.mode}_{datetime.now():%Y%m%d_%H%M%S_%f}"
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

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn



# WGAN-GP flow: noise -> Generator -> synthetic image -> Critic score.
# G maximises the critic score; C learns a real-minus-fake gap with a gradient penalty.
# GENERATOR: random noise [B,100,1,1] -> learned feature maps -> grayscale [B,1,128,128].
# No real MRI is passed into G. The critic supplies the learning signal.
# Transposed convolutions learn upsampling; they do not invert a particular original image.
class Generator(nn.Module):
    def __init__(self, image_size=128, latent_dim=100):
        super().__init__()
        # Input noise [B, latent_dim, 1, 1] becomes a learned [B, 512, 4, 4] map.
        channels, resolution = 512, 4
        layers = [nn.ConvTranspose2d(latent_dim, channels, 4, 1, 0, bias=False),
                  nn.BatchNorm2d(channels), nn.ReLU()]
        # Repeated stride-2 transposed convolutions double H/W and reduce channels.
        while resolution < image_size // 2:
            next_channels = max(32, channels // 2)
            layers.extend([nn.ConvTranspose2d(channels, next_channels, 4, 2, 1, bias=False),
                           nn.BatchNorm2d(next_channels), nn.ReLU()])
            channels, resolution = next_channels, resolution * 2
        # Final grayscale image is [B,1,H,W]. Tanh outputs [-1,1], matching scaled real data.
        layers.extend([nn.ConvTranspose2d(channels, 1, 4, 2, 1), nn.Tanh()])
        self.layers = nn.Sequential(*layers)

    def forward(self, noise):
        return self.layers(noise)


# A critic assigns an unrestricted score, not a real/fake probability.
# No sigmoid, spectral normalisation, weight clipping or batch normalisation:
# the per-image input-gradient penalty supplies the WGAN-GP regularisation.
# CRITIC: image -> progressively smaller feature maps -> one unrestricted score per image.
# Scores are relative, not probabilities: 3.2 is valid and does not mean 320 percent real.
# No BatchNorm keeps each image score independent of other examples for the gradient penalty.
class Critic(nn.Module):
    def __init__(self, image_size=128):
        super().__init__()
        channels, incoming, resolution = 32, 1, image_size
        layers = []
        while resolution > 4:
            layers.extend([nn.Conv2d(incoming, channels, 4, 2, 1), nn.LeakyReLU(0.2)])
            incoming, channels, resolution = channels, min(512, channels*2), resolution // 2
        layers.append(nn.Conv2d(incoming, 1, 4))
        self.layers = nn.Sequential(*layers)

    def forward(self, images):
        return self.layers(images).flatten()

# Convolution weights start as small random values drawn from a normal distribution.
# Random initialisation breaks symmetry: different filters can learn different features.
# START FROM SCRATCH: random convolution weights break symmetry between learned filters.
# This function is applied to submodules only when no saved checkpoint is being loaded.
def initialise(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, mean=0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    # BatchNorm’s learned scale starts near one and its offset starts at zero.
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, mean=1, std=0.02)
        nn.init.zeros_(module.bias)

# Measure how sensitive the critic’s score is to changes in an interpolated image, 
# then penalise gradient norms that differ from one.
# WGAN-GP: encourage the critic input-gradient norm to be near one on interpolated images.
# The gradient is with respect to pixels, then the penalty is differentiated into critic weights.
# This is a soft sampled constraint, not gradient clipping or proof of a global Lipschitz bound.
def gradient_penalty(critic, real, fake):
    # Interpolate independently for each example. Neither endpoint should
    # receive gradients: this penalty trains the critic, not the generator.
    with torch.autocast(device_type=real.device.type, enabled=False):
        # Random alpha in [0,1] for each image, broadcast to all pixels/channels.
        alpha = torch.rand(len(real), 1, 1, 1, device=real.device)
        # Interpolated images require gradients for the penalty; detach real/fake to avoid backprop into G.
        interpolated = (alpha * real.detach().float() +
             +          (1 - alpha) * fake.detach().float()).requires_grad_(True)
        # Each score depends on the mixed image; we measure its sensitivity to every input pixel next.
        scores = critic(interpolated)
        # Compute the gradient of each score with respect to its interpolated image.
        # The returned gradient has the same shape as interpolated: [B, C, H, W].
        gradients = torch.autograd.grad(
            outputs=scores, inputs=interpolated,
            # Weight each output score by one. Retain the derivative graph so GP can train critic weights.
            grad_outputs=torch.ones_like(scores), create_graph=True,
        )[0]
        # Norm across all pixels/channels of EACH image, not across the batch.
        # create_graph=True allows the penalty to backpropagate into critic weights.
        # Preserve B, combine C/H/W, then take one L2 norm per image. Norm 2 gives penalty (2-1)^2=1.
        norms = gradients.flatten(1).norm(2, dim=1)
        return (norms - 1).square().mean(), norms.detach().mean()


# ALTERNATING TRAINING: five real batches update C, then one update changes G (by default).
# C minimises mean(fake score)-mean(real score)+10*GP. G minimises -mean(fake score).
# The two optimisers have separate parameters; each phase controls where gradients can flow.
def train_epoch(generator, critic, loader, args, g_optimizer, c_optimizer, g_scaler):
    generator.train()
    critic.train()
    totals = dict(critic_loss=0., wasserstein_gap=0., gradient_penalty=0.,
                  gradient_norm=0., real_score=0., fake_score=0.)
    count, g_count, g_total, g_updates = 0, 0, 0., 0
    for step, batch in enumerate(loader, 1):
        # Scale real images from [0,1] to [-1,1] to match the generator’s Tanh output.
        real = batch["image"].to(args.device, non_blocking=True).float() * 2 - 1
        batch_size = len(real)
        # Critic-only phase: freeze G, update C to increase real-minus-fake scores while penalising steep input gradients.
        for parameter in critic.parameters():
            parameter.requires_grad_(True)
        c_optimizer.zero_grad(set_to_none=True)
        # G is frozen for critic-only sampling, including its BatchNorm buffers.
        generator.eval()
        # Generate fake images without tracking gradients for G.
        with torch.no_grad(), torch.autocast(device_type=args.device.type, enabled=args.amp):
            fake = generator(torch.randn(batch_size, args.latent_dim, 1, 1, device=args.device))
        generator.train()
        # Keep ALL critic calculations float32, including the second-order GP.

        with torch.autocast(device_type=args.device.type, enabled=False):
            # Compute the mean score for real and fake images, then the gradient penalty.
            real_score = critic(real).mean()
            fake_score = critic(fake.float()).mean()
            gp, norm = gradient_penalty(critic, real, fake)
            gap = real_score - fake_score
            # Minimising -gap increases real-minus-fake scores, while the penalty limits steep input gradients.
            c_loss = -gap + args.gp_weight * gp
        require_finite(c_loss.item())
        # Calculates the critic parameter gradients.
        c_loss.backward()
        # The critic’s parameters are updated to reduce the loss; G is frozen so its weights do not change.
        c_optimizer.step()
        values = dict(critic_loss=c_loss, wasserstein_gap=gap, gradient_penalty=gp,
                      gradient_norm=norm, real_score=real_score, fake_score=fake_score)
        for key, value in values.items():
            totals[key] += value.detach().item() * batch_size
        count += batch_size
        # Each real batch is used once per epoch. Update G after n_critic batches,
        # and once for a final partial group so tiny datasets still train G.
        if step % args.n_critic == 0 or step == len(loader):
            for parameter in critic.parameters():
                parameter.requires_grad_(False)
            # Generator-only phase: freeze C, update G to increase the critic score of its generated images.
            g_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=args.device.type, enabled=args.amp):
                generated = generator(torch.randn(batch_size, args.latent_dim, 1, 1, device=args.device))
            # Do not use no_grad here: gradients must flow THROUGH C into G.
            with torch.autocast(device_type=args.device.type, enabled=False):
                # A higher fake score makes this loss smaller. C stays fixed but its input gradient reaches G.
                g_loss = -critic(generated.float()).mean()
            require_finite(g_loss.item())
            # Calculates the generator parameter gradients by backpropagating through the critic.
            g_scaler.scale(g_loss).backward()
            g_scaler.step(g_optimizer)
            g_scaler.update()
            g_total += g_loss.item() * batch_size
            g_count += batch_size
            g_updates += 1
    for parameter in critic.parameters():
        parameter.requires_grad_(True)
    metrics = {key: value / count for key, value in totals.items()}
    metrics.update(generator_loss=g_total / g_count,
                   critic_updates=len(loader), generator_updates=g_updates)
    for value in metrics.values():
        require_finite(value)
    return metrics


# Evaluation mode fixes BatchNorm statistics; inference_mode disables gradient tracking.
@torch.inference_mode()
def samples(generator, noise, args):
    generator.eval()
    # Generate manageable chunk and convert [-1,1] back to [0,1] for saved images.
    return torch.cat([(generator(chunk) + 1) / 2 for chunk in noise.split(args.batch_size)])


# INSPECTION: Generate images, fixed-noise samples + diversity measurements + a bounded nearest-image preview.
# Pixel diversity may detect repetition, but artifacts can also increase it; inspect images too.
@torch.inference_mode()
def diagnostics(generator, loaders, args, fixed_noise):
    generated = samples(generator, fixed_noise, args)
    image_grid(generated, args.output / "generated_final.png", title="Generated OASIS-style slices")
    real_test = next(iter(loaders["test"]))["image"][:16]
    image_grid(real_test, args.output / "real_test_examples.png", columns=4, title="Held-out real MRI slices")
    # A bounded visual comparison helps inspect obvious copies; it is not a
    # full training-set nearest-neighbour audit or a validated realism metric.
    real_preview = []
    count = 0
    # Collect up to 256 real training slices for a nearest-neighbour comparison.
    for batch in loaders["train"]:
        real_preview.append(batch["image"])
        count += len(batch["image"])
        if count >= 256:
            break
    # Concatenate and move to the selected device for distance calculations.
    real_preview = torch.cat(real_preview)[:256].to(args.device)
    # Downsample before comparing pixels to keep the preview distance matrix small.
    small_fake = F.interpolate(generated, size=(32, 32), mode="area").flatten(1)
    small_real = F.interpolate(real_preview, size=(32, 32), mode="area").flatten(1)
    # Squared Euclidean distance divided by pixel count equals per-image-pair MSE.
    distances = torch.cdist(small_fake, small_real).square() / small_fake.size(1)
    nearest_error, nearest_index = distances.min(1)
    pairs = torch.stack((generated[:8], real_preview[nearest_index[:8]]), dim=1).flatten(0, 1)
    image_grid(pairs, args.output / "nearest_training_preview.png", columns=4,
               title="Alternating generated / nearest of up to 256 training slices (32x32 MSE)")
    # Compare generated images with each other: very small distances can suggest
    # mode collapse, where different noise vectors produce nearly identical images.
    pairwise = torch.pdist(small_fake).square() / small_fake.size(1)
    report = dict(generated_samples=len(generated),
                # The mean pixel standard deviation is a rough measure of diversity across generated images.
                  mean_pixel_standard_deviation=float(generated.std(dim=0).mean().item()),
                  mean_pairwise_mse_32x32=float(pairwise.mean().item()),
                  mean_nearest_training_preview_mse_32x32=float(nearest_error.mean().item()),
                  reference_training_slices=len(real_preview),
                  note="Small diversity or repeated images may indicate mode collapse. These numbers do not establish realism, novelty or absence of memorisation.")
    save_json(args.output / "generation_diagnostics.json", report)
    np.savez_compressed(args.output / "generated_samples.npz", images=generated.cpu().numpy(),
                        noise=fixed_noise.cpu().numpy())
    return report


# DEMO ROUTE: build G/C -> initialise or load -> alternate updates -> save samples/checkpoint.
# There is no best-validation-realism selection: last.pt is the final saved training epoch.
def main():
    parser = parser_for(__doc__, epochs=100, batch_size=32, lr=1e-4)
    parser.set_defaults(amp=False)
    parser.add_argument("--n-critic", type=int, default=5, help="Critic batches per G update; final partial group also updates G")
    parser.add_argument("--gp-weight", type=float, default=10., help="Weight of input-gradient norm penalty")
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--sample-every", type=int, default=5)
    args, checkpoint = configure(parser, "gan")
    if args.latent_dim < 1 or args.sample_every < 1:
        parser.error("latent-dim and sample-every must be positive.")
    loaders, split, _ = make_loaders(args, checkpoint=checkpoint)

    # Build the generator and critic, then either initialise or load weights.
    generator = Generator(args.image_size, args.latent_dim).to(args.device)
    critic = Critic(args.image_size).to(args.device)
    # A separate seeded preview RNG avoids consuming random numbers used by training.
    fixed_rng = torch.Generator(device=args.device).manual_seed(args.seed + 1000)
    # Reusing the same 64 noise vectors makes visual changes across epochs comparable.
    fixed_noise = torch.randn(64, args.latent_dim, 1, 1, generator=fixed_rng, device=args.device)
    training_seconds = None
    if checkpoint:
        generator.load_state_dict(checkpoint["generator"])
        critic.load_state_dict(checkpoint["critic"])
        fixed_noise = checkpoint["fixed_noise"].to(args.device)
    else:
        generator.apply(initialise)
        critic.apply(initialise)
        # Separate Adam optimisers track different parameters/momentum for the two networks.
        # Adam betas control first/second moment averaging, not the critic-to-generator update ratio.
        g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0, 0.9))
        c_optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr, betas=(0.0, 0.9))
        g_scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
        image_grid(next(iter(loaders["train"]))["image"][:16], args.output / "real_training_examples.png", columns=4)
        history = []
        sync(args.device)
        started = time.perf_counter()
        # Alternate updates for the requested number of epochs. Each epoch uses every training batch once.
        for number in range(1, args.epochs+1):
            metrics = train_epoch(generator, critic, loaders["train"], args,
                                  g_optimizer, c_optimizer, g_scaler)
            history.append(dict(epoch=number, **metrics))
            # Save the latest epoch. GAN loss alone does not establish the best image quality.
            save_model(args.output / "last.pt", "gan", args, split, number,
                       algorithm="wgan-gp", generator=generator.state_dict(), critic=critic.state_dict(),
                       fixed_noise=fixed_noise.cpu())
            if number == 1 or number % args.sample_every == 0 or number == args.epochs:
                image_grid(samples(generator, fixed_noise, args), args.output / f"generated_epoch_{number:04d}.png",
                           title=f"Fixed-noise samples after epoch {number}")
            save_history(args.output, history, ["generator_loss", "critic_loss", "wasserstein_gap", "gradient_penalty"])
            print(f"Epoch {number}/{args.epochs}: G={metrics['generator_loss']:.4f}, "
                  f"C={metrics['critic_loss']:.4f}, gap={metrics['wasserstein_gap']:.4f}, "
                  f"GP={metrics['gradient_penalty']:.4f}, norm={metrics['gradient_norm']:.3f}", flush=True)
        sync(args.device)
        training_seconds = time.perf_counter() - started
    # This GAN has no validation-based checkpoint selection. Test images appear
    # only as a held-out visual reference in the final diagnostics.
    report = diagnostics(generator, loaders, args, fixed_noise)
    report.update(algorithm="wgan-gp", n_critic=args.n_critic, gp_weight=args.gp_weight,
                  training_seconds=training_seconds, image_size=args.image_size,
                  slices_per_volume=args.slices_per_volume,
                  checkpoint_selection="Last training epoch; no automatic claim of best realism.")
    save_json(args.output / "results.json", report)
    print("Diagnostics:", report)
    print("Results:", args.output.resolve())


# Run training/evaluation only when launched directly, not when imported.
# This guard also matters when DataLoader starts worker processes on Windows.
if __name__ == "__main__":
    main()
