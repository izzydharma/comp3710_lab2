"""Part 4 Task 2: categorical UNet segmentation of OASIS brain MRI.

Click Run to train using the paired MRI/mask PNG folders beside this file.
    python task2_unet.py --epochs 100
    python task2_unet.py --mode evaluate --checkpoint PATH_TO_BEST_PT

Evaluation loads the trained model, segments held-out subjects, reports
Dice for every label and saves MRI / ground truth / prediction figures.
All slices are used by default at the supplied PNG resolution (256x256).
Changing --image-size changes the evaluation resolution. These are 2D slice
scores, not original 3D volume scores. No pretrained model or local helper modules.
Source: https://arxiv.org/abs/1505.04597
ChatGPT assisted with implementation. Use --data-dir PATH if the extracted PNG dataset is elsewhere.

Mask values 0, 85, 170, 255 map to categorical channels 0, 1, 2, 3.
Raw label 0 is treated as background. Anatomical names must be checked against
the course's label definitions. Case IDs never overlap across data splits.
Defaults: 100 epochs, batch size 8. Outputs go into part4_results.
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

    # Dataset length counts individual 2D slices, not the number of subjects.
    def __len__(self):
        return len(self.rows)

    # Load one aligned image/mask pair lazily; the DataLoader forms batches later.
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
from torch import nn



# Two 3x3 convolutions extract features while padding preserves H and W.
# GroupNorm normalises within each sample, avoiding dependence on batch statistics.
# LOCAL FEATURES: two 3x3 convolutions build features while padding preserves height/width.
# GroupNorm uses groups within each sample, so it works with the small training batch.
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        # Eight groups share the channels within each sample. Unlike BatchNorm,
        # GroupNorm has no running batch statistics; it also learns scale and offset.
        nn.GroupNorm(8, out_channels), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(8, out_channels), nn.ReLU(),
    )


# U-Net predicts a category for every pixel: [B, 1, H, W] -> [B, classes, H, W].
# U-NET ARCHITECTURE: encoder shrinks maps; decoder restores resolution with encoder skips.
# Input [B,1,H,W] -> logits [B,4,H,W]. This predicts a category at every pixel.
# Skips concatenate features along channels.
class UNet(nn.Module):
    def __init__(self, classes, base_channels=32):
        super().__init__()
        # Default channels: 32, 64, 128, 256, 512. Four pooling steps shrink 256 -> 16.
        widths = [base_channels * (2**i) for i in range(5)]
        # ModuleList registers parameters for optimisation; forward explicitly loops
        # over these layers. It does not automatically apply them like Sequential.
        # Each double_conv preserves spatial size while increasing channel count.
        self.encoders = nn.ModuleList([double_conv(1, widths[0])] +
                                     [double_conv(widths[i-1], widths[i]) for i in range(1, 5)])
        # The bottleneck is the deepest encoder output; it is the first decoder input.
        # Each transposed convolution doubles resolution and reduces channel count.
        self.up = nn.ModuleList([nn.ConvTranspose2d(widths[i+1], widths[i], 2, 2)
                                 for i in range(3, -1, -1)])
        # Concatenating a skip and an upsampled map doubles the incoming channels.
        self.decoders = nn.ModuleList([double_conv(2 * widths[i], widths[i]) for i in range(3, -1, -1)])
        # A 1x1 convolution converts each pixel feature vector into class logits.
        self.output = nn.Conv2d(widths[0], classes, 1)

    def forward(self, images):
        # For default 256px input, encoder maps are:
        # 32x256x256 -> 64x128x128 -> 128x64x64 -> 256x32x32 -> 512x16x16.
        # The batch dimension B remains unchanged throughout.
        saved, hidden = [], images
        for index, encoder in enumerate(self.encoders):
            if index:
                # Take the maximum in each 2x2 window, halving height and width.
                hidden = F.max_pool2d(hidden, 2)
            hidden = encoder(hidden)
            # Save encoder detail at each resolution for the matching decoder stage.
            saved.append(hidden)
        # Traverse from the bottleneck back to full resolution, using skips in reverse order.
        # saved[:-1] excludes the bottleneck: it is already the decoder input.
        # First decoder step: upsample to 256x32x32, concatenate the matching
        # 256-channel skip to get 512 channels, then double_conv returns 256.
        for up, decoder, skip in zip(self.up, self.decoders, reversed(saved[:-1])):
            hidden = up(hidden)
            # Align spatial sizes before concatenation if rounding caused a mismatch.
            if hidden.shape[-2:] != skip.shape[-2:]:
                hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            # dim=1 joins channels: fine encoder detail + contextual decoder features.
            hidden = decoder(torch.cat((skip, hidden), dim=1))
        return self.output(hidden)  # One logit channel per segmentation category.


# OBJECTIVE: cross entropy rewards the correct pixel class; soft Dice rewards region overlap.
# Soft probabilities keep Dice differentiable. Hard argmax masks are used later for reporting.
def segmentation_loss(logits, labels):
    # Integer masks [B,H,W] become one-hot targets [B,C,H,W] for C categories.
    # Example: class ID 2 becomes [0,0,1,0]. permute moves classes to the channel axis.
    one_hot = F.one_hot(labels, num_classes=logits.size(1)).permute(0, 3, 1, 2).float()
    # Compute loss reductions in float32 even when model inference uses AMP.
    logits = logits.float()
    # At each pixel, softmax makes the four category probabilities sum to one.
    # These probabilities are used for Dice; cross entropy below gets raw logits.
    probabilities = logits.softmax(dim=1)
    # Sum over batch and spatial axes, leaving one soft Dice score per class.
    axes = (0, 2, 3)
    # Only probability assigned to the true class contributes to that class intersection.
    intersection = (probabilities * one_hot).sum(axes)
    denominator = probabilities.sum(axes) + one_hot.sum(axes)
    # Soft Dice uses probabilities so it remains differentiable; epsilon avoids 0/0.
    # Cross entropy rewards correct pixel classes; Dice rewards region overlap.
    # This loss averages all classes, including background.
    # The small epsilon prevents division by zero; this score is computed for each class.
    soft_dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
    # Categorical one-hot target, not binary/regression segmentation.
    # PyTorch accepts floating one-hot targets here. Both terms have weight 1.
    # Hard argmax masks are not used in this loss because argmax blocks learning gradients.
    return F.cross_entropy(logits, one_hot) + (1 - soft_dice.mean())


# METRIC: Dice = 2TP/(2TP+FP+FN). Perfect overlap gives 1; missed/extra pixels lower it.
# If a class is absent in both masks there is no evidence to score, so return NaN.
def dice_scores(confusion):
    confusion = np.asarray(confusion, dtype=np.float64)
    # Rows are true classes, columns predicted classes. The diagonal counts true positives.
    truth = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    denominator = truth + predicted
    # Dice = 2TP/(truth count + prediction count). No truth/prediction gives NaN, not 1.
    return np.divide(2 * np.diag(confusion), denominator,
                     out=np.full(len(truth), np.nan), where=denominator > 0)


# EPOCH WORKFLOW: paired MRI/mask -> intensity augmentation if training -> logits -> loss/update.
# Then count true/predicted pixel pairs for global and, during evaluation, per-subject metrics.
def epoch(model, loader, args, classes, optimizer=None, scaler=None):
    # The same loop trains when an optimiser is supplied, otherwise only evaluates.
    training = optimizer is not None
    model.train(training)
    total_loss, count = 0.0, 0
    confusion = torch.zeros((classes, classes), dtype=torch.long, device=args.device)
    subjects = {}
    with torch.set_grad_enabled(training):
        for batch in loader:
            images = batch["image"].to(args.device, non_blocking=True)
            labels = batch["mask"].to(args.device, non_blocking=True)
            if training:
                # Intensity augmentation preserves the image/mask spatial alignment.
                gain = torch.empty(len(images), 1, 1, 1, device=args.device).uniform_(0.9, 1.1)
                images = (images * gain).clamp(0, 1)
                # Discard old gradients: PyTorch otherwise accumulates them across batches.
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=args.device.type, enabled=args.amp):
                logits = model(images)
                loss = segmentation_loss(logits, labels)
            if training:
                # AMP gradient scaling protects small gradients before the optimiser update.
                # backward computes gradients; step updates weights; update adjusts AMP scaling.
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            # Hard argmax labels are for reporting only; the loss used probabilities/logits.
            # Choose the largest of four channel scores at every pixel: [B,4,H,W] -> [B,H,W].
            predicted = logits.detach().argmax(1)
            # Report a sample-weighted average of batch losses. The soft Dice component
            # is computed per batch, so this is not a global pooled-Dice loss.
            total_loss += loss.item() * len(images)
            count += len(images)
            for index, subject in enumerate(batch["subject_id"]):
                # Encode (true, predicted) as true*C + predicted, count pixels, then reshape.
                # This builds a C-by-C confusion matrix for the current slice.
                matrix = torch.bincount((labels[index] * classes + predicted[index]).flatten(),
                                        minlength=classes**2).reshape(classes, classes)
                confusion += matrix
                if not training:
                    if subject not in subjects:
                        subjects[subject] = np.zeros((classes, classes), dtype=np.int64)
                    # Aggregate all evaluated slices of this case before computing its Dice.
                    subjects[subject] += matrix.cpu().numpy()
    require_finite(total_loss / count)
    matrix = confusion.cpu().numpy()
    return dict(loss=total_loss/count, confusion=matrix, subjects=subjects)


# TWO AGGREGATIONS: pooled Dice combines pixels; mean-subject Dice averages case scores.
# These answer different questions and need not match, even for the same predictions.
def report_dice(result, class_values):
    # Pooled Dice combines all pixels. Mean subject Dice gives each evaluable case
    # equal weight, so it can differ from the score dominated by larger regions/cases.
    pooled = dice_scores(result["confusion"])
    subject_scores = {subject: dice_scores(matrix) for subject, matrix in result["subjects"].items()}
    rows = []
    for index, value in enumerate(class_values):
        available = [scores[index] for scores in subject_scores.values() if np.isfinite(scores[index])]
        rows.append(dict(raw_label=value, pooled_dice=float(pooled[index]) if np.isfinite(pooled[index]) else None,
                         mean_subject_dice=float(np.mean(available)) if available else None,
                         ground_truth_pixels=int(result["confusion"][index].sum()),
                         evaluated_subjects=len(available)))
    # An absent class must not be counted as a perfect score or a passed target.
    # This Boolean checks pooled and mean-subject Dice for every label.
    # It does not itself require every individual subject score to exceed 0.9.
    passed = all(row["ground_truth_pixels"] > 0 and row["pooled_dice"] is not None and
                 row["pooled_dice"] > 0.9 and row["mean_subject_dice"] is not None and
                 row["mean_subject_dice"] > 0.9 for row in rows)
    return dict(labels=rows, all_labels_above_0_9=passed,
                per_subject={subject: {str(value): float(scores[i]) if np.isfinite(scores[i]) else None
                                      for i, value in enumerate(class_values)}
                             for subject, scores in subject_scores.items()},
                convention="Dice=2TP/(2TP+FP+FN); absent in prediction and truth => null. "
                           "Pooled and mean-per-subject scores both reported, including background.")


# DISPLAY: compare MRI, ground truth and argmax prediction at the same slice location.
# Matching colours represent matching classes, not matching grayscale intensity values.
@torch.inference_mode()
def visualise(model, loader, args, class_values):
    model.eval()
    # Select visible brain examples for the figure only; metrics above use all slices.
    chosen = []
    background = class_values.index(0) if 0 in class_values else -1
    for incoming in loader:
        for i in range(len(incoming["image"])):
            if (incoming["mask"][i] != background).float().mean() > 0.01:
                chosen.append({key: value[i] for key, value in incoming.items()})
            if len(chosen) == 6:
                break
        if len(chosen) == 6:
            break
    if not chosen:
        raise ValueError("No foreground MRI slices available for the segmentation figure.")
    batch = dict(image=torch.stack([row["image"] for row in chosen]),
                 mask=torch.stack([row["mask"] for row in chosen]),
                 subject_id=[row["subject_id"] for row in chosen],
                 slice_index=torch.stack([row["slice_index"] for row in chosen]))
    images = batch["image"].to(args.device)
    labels = batch["mask"]
    # Logits/probabilities: [B,4,H,W]. argmax below gives [B,H,W] class IDs.
    # A one-hot prediction then restores four binary channels for saving.
    probabilities = model(images).float().softmax(1).cpu()
    predicted = probabilities.argmax(1)
    # Save both hard one-hot masks and soft probabilities for later inspection.
    one_hot = F.one_hot(predicted, len(class_values)).permute(0, 3, 1, 2).to(torch.uint8)
    fig, axes = plt.subplots(len(images), 3, figsize=(9, 3*len(images)), squeeze=False)
    for i in range(len(images)):
        axes[i, 0].imshow(images[i, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[i, 1].imshow(labels[i], cmap="tab20", vmin=0, vmax=len(class_values)-1)
        axes[i, 2].imshow(predicted[i], cmap="tab20", vmin=0, vmax=len(class_values)-1)
        for ax, title in zip(axes[i], ("MRI", "Ground truth", "Prediction")):
            ax.set_title(f"{title}: {batch['subject_id'][i]}, slice {int(batch['slice_index'][i])}", fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.output / "segmentations.png", dpi=150)
    plt.close(fig)
    np.savez_compressed(args.output / "segmentation_examples.npz",
                        one_hot=one_hot.numpy(), probabilities=probabilities.numpy(),
                        raw_prediction=np.asarray(class_values)[predicted.numpy()],
                        raw_truth=np.asarray(class_values)[labels.numpy()],
                        class_values=np.asarray(class_values),
                        subject_ids=np.asarray(batch["subject_id"][:len(images)], dtype=str),
                        slice_indices=batch["slice_index"][:len(images)].numpy())


# DEMO ROUTE: pair data -> train U-Net -> select validation foreground Dice -> test selected weights.
# Background is excluded from selection but included in the training loss and final label report.
def main():
    parser = parser_for(__doc__, epochs=100, batch_size=8, lr=1e-3, slices=0)
    parser.set_defaults(image_size=256)  # Native resolution of the supplied PNG slices.
    parser.add_argument("--base-channels", type=int, default=32)
    args, checkpoint = configure(parser, "unet")
    if args.base_channels < 8 or args.base_channels % 8:
        parser.error("base-channels must be a positive multiple of eight.")
    loaders, split, class_values = make_loaders(args, segmentation=True, checkpoint=checkpoint)
    print("Channel to raw label mapping:", dict(enumerate(class_values)), flush=True)
    # Four output categories come from MASK_VALUES, including background.
    # This is supervised segmentation: targets are masks, not reconstructed MRI pixels.
    model = UNet(len(class_values), args.base_channels).to(args.device)
    training_seconds = None
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        # Adam adapts updates using moving gradient and squared-gradient estimates.
        # Defaults here: lr=0.001, 100 epochs, batch size 8; no learning-rate scheduler.
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
        best, history = -1.0, []
        sync(args.device)
        started = time.perf_counter()
        for number in range(1, args.epochs+1):
            train = epoch(model, loaders["train"], args, len(class_values), optimizer, scaler)
            val = epoch(model, loaders["val"], args, len(class_values))
            scores = dice_scores(val["confusion"])
            # Select on mean foreground Dice, assuming raw label 0 is background.
            indices = [i for i, value in enumerate(class_values) if value != 0 and np.isfinite(scores[i])]
            if not indices:
                raise ValueError("Validation contains no evaluable foreground labels.")
            # Average the pooled validation Dice across foreground classes only.
            # This selection score differs from the all-class soft Dice used in training.
            score = float(np.mean(scores[indices]))
            history.append(dict(epoch=number, train_loss=train["loss"], val_loss=val["loss"], val_dice=score))
            # Keep the highest validation foreground Dice, not the lowest training loss.
            if score > best:
                best = score
                save_model(args.output / "best.pt", "unet", args, split, number,
                           model=model.state_dict(), class_values=class_values, validation_dice=score)
            save_history(args.output, history, ["train_loss", "val_loss", "val_dice"])
            print(f"Epoch {number}/{args.epochs}: train={train['loss']:.4f}, val={val['loss']:.4f}, "
                  f"val foreground DSC={score:.4f}", flush=True)
        sync(args.device)
        training_seconds = time.perf_counter() - started
        checkpoint = torch.load(args.output / "best.pt", map_location=args.device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
    sync(args.device)
    started = time.perf_counter()
    # Test only after checkpoint selection; these labels were not used to train/tune weights.
    result = epoch(model, loaders["test"], args, len(class_values))
    sync(args.device)
    report = report_dice(result, class_values)
    # This duration includes the evaluation loop, loss/metric work and reporting
    # so far; it is not isolated single-image network latency.
    report.update(test_loss=result["loss"], inference_seconds=time.perf_counter()-started,
                  training_seconds=training_seconds, selected_epoch=checkpoint["epoch"],
                  evaluation=f"Held-out cases, {args.image_size}x{args.image_size} 2D slices.",
                  slices_per_volume=args.slices_per_volume)
    save_json(args.output / "dice_results.json", report)
    visualise(model, loaders["test"], args, class_values)
    for row in report["labels"]:
        print(f"Raw label {row['raw_label']}: pooled DSC={row['pooled_dice']}, "
              f"mean subject DSC={row['mean_subject_dice']}")
    print("All labels above 0.9:", report["all_labels_above_0_9"])
    print("Results:", args.output.resolve())


# Run training/evaluation only when launched directly, not when imported.
# This guard also matters when DataLoader starts worker processes on Windows.
if __name__ == "__main__":
    main()
