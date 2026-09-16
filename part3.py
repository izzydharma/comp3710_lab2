"""COMP3710 Lab 2, Part 3.2: CIFAR-10 ResNet-18. Part 3.1 is in lab2.ipynb.

Run from the folder containing this file:
    python part3.py cifar10 --device cuda --epochs 100
    python part3.py demo --checkpoint PATH_TO_BEST_PT --device cuda

Dependencies: numpy, matplotlib, scikit-learn, torch, torchvision.
Use the official PyTorch installer to match torch/torchvision and CUDA:
https://pytorch.org/get-started/locally/

Assignment requirement: Part 3.2 uses ResNet-18 trained from scratch. The >90% and 94%/360-second
targets require actual experiments; this script does not guarantee them.
The cluster demonstration must still be performed on Rangpur.

CUDA and mixed precision are enabled by default. GPU runs cache all splits as
uint8 images, augment batches and calculate evaluation metrics on-device.
Use --device cpu for CPU execution or --no-gpu-data for CPU data loaders.
Cache setup is timed. File loading/splitting, logging and plotting use the CPU.

Sources: COMP3710 Lab 2, Part 3.2.
ChatGPT assisted with implementation and explanations.
"""

import argparse
import csv
import json
import math
import os
import platform
import random
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save plots on both laptops and headless cluster nodes.
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset




# Part 3.2: basic residual block, implemented directly (no pre-built models).
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Learn a residual correction with two 3x3 convolutions.
        # Batch normalisation follows each convolution, so convolution bias is unnecessary.
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # A projection aligns dimensions when spatial size/channels change.
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        # Add the shortcut to the learned correction. This also provides a direct
        # path for gradients, helping deeper networks train.
        return torch.relu(self.main(x) + self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        # CIFAR-10 has 32x32 images: use a 3x3 stride-1 stem without max pooling.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        stages = []
        in_channels = 64
        # Four stages, with two residual blocks each: 8 blocks x 2 convolutions.
        # Together with the stem and final linear layer, these give the usual 18-layer count.
        # Stage outputs have spatial sizes 32x32, 16x16, 8x8 and 4x4.
        for index, channels in enumerate((64, 128, 256, 512)):
            stages.extend([
                BasicBlock(in_channels, channels, stride=1 if index == 0 else 2),
                BasicBlock(channels, channels),
            ])
            in_channels = channels
        self.stages = nn.Sequential(*stages)
        # Average each final feature map to one number: 512 features per image.
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Produce one raw class score (logit) for each of the 10 CIFAR-10 classes.
        self.fc = nn.Linear(512, 10)
        # Initialise convolution weights for ReLU; training starts from scratch.
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.fc(self.pool(self.stages(self.stem(x))).flatten(1))


def seed_worker(_):
    # Give NumPy and Python random operations a seed in each DataLoader worker.
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


class CachedCifarBatches:
    """Keep compact uint8 images on the GPU and transform whole batches.

    Uses the same zero-padded random crop, horizontal flip and normalisation
    as the torchvision pipeline. Random draws differ, but the distributions
    and held-out split are unchanged. No pretrained weights are used.
    """
    def __init__(self, dataset, indices, args, mean, std, training):
        self.dataset = Subset(dataset, indices)
        self.batch_size = args.batch_size
        self.training = training
        self.device = args.device
        # Copy only this split to the device once. uint8 uses one byte per pixel
        # value, reducing cache memory compared with float32. Raw layout is NHWC.
        self.images = torch.from_numpy(dataset.data[indices]).to(self.device)
        self.labels = torch.as_tensor(np.asarray(dataset.targets)[indices],
                                      dtype=torch.long, device=self.device)
        if training:
            # Pad raw pixels with zero BEFORE normalising, matching RandomCrop.
            self.images = torch.nn.functional.pad(
                self.images.permute(0, 3, 1, 2), (4, 4, 4, 4)
            ).permute(0, 2, 3, 1).contiguous()
        # Broadcast the three RGB statistics across every image, row and column.
        self.mean = torch.tensor(mean, device=self.device).view(1, 1, 1, 3)
        self.std = torch.tensor(std, device=self.device).view(1, 1, 1, 3)
        # A separate random generator controls shuffling, crops and flips.
        self.generator = torch.Generator(device=self.device).manual_seed(args.seed)
        self.axis = torch.arange(32, device=self.device)

    def __len__(self):
        # Round up so the final, smaller batch is included.
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        # Shuffle training examples each epoch; keep evaluation order fixed.
        if self.training:
            order = torch.randperm(len(self.dataset), device=self.device,
                                   generator=self.generator)
        else:
            order = torch.arange(len(self.dataset), device=self.device)
        for start in range(0, len(self.dataset), self.batch_size):
            indices = order[start:start + self.batch_size]
            if self.training:
                count = indices.numel()
                # Padding makes a 40x40 image. A 32x32 crop can start at offsets 0 through 8.
                # Choose independent vertical and horizontal offsets for each image.
                offsets = torch.randint(9, (count, 2), device=self.device,
                                        generator=self.generator)
                # Flip each training image horizontally with probability 0.5.
                flip = torch.rand(count, device=self.device,
                                  generator=self.generator) < 0.5
                rows = offsets[:, 0, None] + self.axis[None, :]
                cols = offsets[:, 1, None] + torch.where(
                    flip[:, None], 31 - self.axis[None, :], self.axis[None, :]
                )
                # The None dimensions broadcast image, row and column indices to gather
                # a whole batch of 32x32 RGB crops in one operation on the device.
                pixels = self.images[indices[:, None, None], rows[:, :, None],
                                     cols[:, None, :], :]
            else:
                pixels = self.images[indices]
            # Convert 0..255 pixels to 0..1, then standardise each RGB channel.
            # Standardised values do not have to remain between 0 and 1.
            pixels = (pixels.float().div_(255.0) - self.mean) / self.std
            # NHWC contiguous -> NCHW view with channels-last physical layout.
            yield pixels.permute(0, 3, 1, 2), self.labels[indices]


def load_data(task, args):
    """Keep test data out of training, tuning, and checkpoint selection."""
    args.data_dir.mkdir(parents=True, exist_ok=True)
    from torchvision import datasets as vision_datasets, transforms
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    # Training augmentation varies the images to help generalisation.
    # Validation and test images receive only conversion and normalisation.
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    train_full = vision_datasets.CIFAR10(
        str(args.data_dir), train=True, download=args.download, transform=train_transform,
    )
    # Use the same source training images with evaluation transforms for validation.
    val_full = vision_datasets.CIFAR10(
        str(args.data_dir), train=True, download=False, transform=eval_transform,
    )
    # Hold out 5,000 of the 50,000 training images for validation.
    # Stratification preserves class proportions; the official test set stays separate.
    train_idx, val_idx = train_test_split(
        np.arange(len(train_full)), test_size=0.1,
        stratify=train_full.targets, random_state=args.seed,
    )
    test = vision_datasets.CIFAR10(
        str(args.data_dir), train=False, download=args.download, transform=eval_transform,
    )
    datasets = [Subset(train_full, train_idx), Subset(val_full, val_idx), test]
    spec = dict(task=task, classes=list(train_full.classes), mean=list(mean), std=list(std))
    args.gpu_data_setup_seconds = 0.0
    # Caching avoids repeated CPU transforms and host-to-GPU copies each epoch.
    # Measure cache setup separately so it can be included in the training budget.
    if args.gpu_data and args.device.type == "cuda":
        sync(args.device)
        cache_started = time.perf_counter()
        loaders = [
            CachedCifarBatches(train_full, train_idx, args, mean, std, True),
            CachedCifarBatches(val_full, val_idx, args, mean, std, False),
            CachedCifarBatches(test, np.arange(len(test)), args, mean, std, False),
        ]
        sync(args.device)
        args.gpu_data_setup_seconds = time.perf_counter() - cache_started
        print("GPU data cache enabled; batched crop/flip/normalisation. "
              "DataLoader workers are not used in this mode.", flush=True)
    else:
        # Fallback pipeline: workers prepare batches on the CPU. Pinned memory
        # supports faster CUDA transfers; persistent workers avoid restarting each epoch.
        loaders = [DataLoader(
            dataset, batch_size=args.batch_size, shuffle=(i == 0),
            num_workers=args.workers, pin_memory=args.device.type == "cuda",
            persistent_workers=args.workers > 0, worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(args.seed + i),
        ) for i, dataset in enumerate(datasets)]
    print("Train / validation / test:", *(len(data) for data in datasets), flush=True)
    return loaders, spec


def make_model(spec):
    return ResNet18()


def sync(device):
    # CUDA launches work asynchronously. Wait for completion before reading timers.
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_epoch(model, loader, device, amp, optimizer=None, scaler=None, collect=False):
    # One function handles both training and evaluation. BatchNorm updates its
    # running statistics in training mode and uses stored statistics in evaluation.
    training = optimizer is not None
    model.train(training)
    # Accumulate metrics on the device, avoiding CPU synchronisation every batch.
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), dtype=torch.long, device=device)
    total = 0
    truth, predictions = [], []
    # Cross entropy accepts raw logits and integer class labels.
    # Do not apply softmax first: the loss handles log-softmax internally.
    criterion = nn.CrossEntropyLoss()
    # Evaluation needs predictions, but no gradient graph or weight updates.
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            if training:
                # Clear old gradients before this batch; PyTorch otherwise accumulates them.
                optimizer.zero_grad(set_to_none=True)
            # Mixed precision uses lower precision for suitable operations to speed up CUDA.
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                # Backpropagation computes gradients. Scaling helps prevent small float16
                # gradients from underflowing; step applies the optimiser update when safe.
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            # The largest class score is the prediction. Detach metrics from the gradient graph.
            predicted = logits.detach().argmax(1)
            total += labels.size(0)
            # Weight each batch mean by its size so the smaller last batch counts correctly.
            total_loss += loss.detach().float() * labels.size(0)
            total_correct += (predicted == labels).sum()
            # Keep report inputs on-device too; do not synchronise each test batch.
            if collect:
                truth.append(labels.detach())
                predictions.append(predicted)
    mean_loss, accuracy = torch.stack((total_loss / total, total_correct / total)).tolist()
    if not math.isfinite(mean_loss):
        raise RuntimeError("Non-finite loss; try a lower learning rate or --no-amp.")
    return dict(loss=mean_loss, accuracy=accuracy,
                truth=torch.cat(truth) if collect else None,
                predictions=torch.cat(predictions) if collect else None)


def evaluation_metrics(truth, predictions, classes):
    """Calculate the confusion matrix and report metrics on the input device."""
    count = len(classes)
    matrix = torch.bincount(truth * count + predictions,
                            minlength=count * count).reshape(count, count)
    support = matrix.sum(1)
    true_positive = matrix.diag().double()
    precision = true_positive / matrix.sum(0).clamp_min(1)
    recall = true_positive / support.clamp_min(1)
    f1 = 2 * true_positive / (matrix.sum(0) + support).clamp_min(1)
    scores = torch.stack((precision, recall, f1), dim=1)
    total = support.sum()
    macro = scores.mean(0)
    weighted = (scores * support[:, None]).sum(0) / total.clamp_min(1)
    accuracy = true_positive.sum() / total.clamp_min(1)
    # Transfer only completed results for text formatting and matplotlib.
    rows = torch.cat((
        torch.cat((scores, support[:, None]), dim=1),
        torch.cat((macro, total[None]))[None, :],
        torch.cat((weighted, total[None]))[None, :],
    )).cpu().tolist()
    lines = [f"{'':>14} {'precision':>9} {'recall':>9} {'f1-score':>9} {'support':>9}", ""]
    for name, (p, r, f, n) in zip([*classes, 'macro avg', 'weighted avg'], rows):
        lines.append(f"{name:>14} {p:9.2f} {r:9.2f} {f:9.2f} {int(n):9d}")
    lines.extend(("", f"Accuracy: {accuracy.item():.4f}"))
    return matrix.cpu().numpy(), "\n".join(lines) + "\n"


def save_history(history, output):
    # Save measurements as a table and plot learning progress. Missing validation
    # values mean validation was skipped that epoch, not that accuracy was zero.
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ("loss", "accuracy")):
        for split in ("train", "val"):
            ax.plot([r["epoch"] for r in history], [r[f"{split}_{metric}"] for r in history], label=split)
        ax.set(xlabel="Epoch", ylabel=metric.title())
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=150)
    plt.close(fig)


def save_evaluation(model, loader, spec, args, result):
    classes = spec["classes"]
    # Precision, recall and F1 show performance per class; the confusion matrix
    # shows which true classes are being mistaken for other classes.
    matrix, report = evaluation_metrics(result["truth"], result["predictions"], classes)
    print(report)
    (args.output / "classification_report.txt").write_text(report, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(matrix, display_labels=classes).plot(
        ax=ax, xticks_rotation=90, colorbar=False,
    )
    fig.tight_layout()
    fig.savefig(args.output / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    images, truth = next(iter(loader))
    images, truth = images[:12], truth[:12]
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type=args.device.type, enabled=args.amp):
        predicted = model(images.to(args.device)).argmax(1).cpu()
    # Undo normalisation on the compute device before copying pixels for plotting.
    shown = images.detach().to(args.device).clone()
    truth = truth.cpu()
    if spec["task"] == "cifar10":
        shown = shown * torch.tensor(spec["std"], device=args.device)[None, :, None, None]
        shown += torch.tensor(spec["mean"], device=args.device)[None, :, None, None]
    shown = shown.clamp_(0, 1).cpu()
    fig, axes = plt.subplots(3, 4, figsize=(13, 8))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i >= len(shown):
            continue
        image = shown[i].permute(1, 2, 0).numpy()
        ax.imshow(image[..., 0] if image.shape[-1] == 1 else image, cmap="gray")
        ax.set_title(f"True: {classes[int(truth[i])]}\nPred: {classes[int(predicted[i])]}",
                     fontsize=9, color="green" if truth[i] == predicted[i] else "red")
    fig.tight_layout()
    fig.savefig(args.output / "predictions.png", dpi=150)
    plt.close(fig)


def main():
    # Defaults allow the Run button to start training. Command-line flags can
    # change settings or select demo mode without editing this file.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", choices=["cifar10", "demo"], nargs="?", default="cifar10")
    parser.add_argument("--checkpoint", type=Path, help="Saved best.pt; required for demo")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--output", type=Path, help="New/empty result directory")
    parser.add_argument("--epochs", type=int, help="Default: 100; demo always 1")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--val-every", type=int, default=5,
                        help="Validate and checkpoint every N epochs; the final epoch is always validated.")
    parser.add_argument("--lr", type=float,
                        help="Training default scales from 0.1 at batch size 128; demo default is 0.001")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0 if os.name == "nt" else 4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-data", action=argparse.BooleanOptionalAction, default=True,
                        help="Cache CIFAR-10 on CUDA and augment batches there; --no-gpu-data restores CPU loaders.")
    args = parser.parse_args()
    if (args.batch_size < 1 or args.val_every < 1 or args.workers < 0
            or (args.epochs is not None and args.epochs < 1)):
        parser.error("Batch size/validation interval/epochs must be positive and workers nonnegative.")
    if args.lr is not None and args.lr <= 0:
        parser.error("Learning rate must be positive.")
    if args.task == "demo" and args.checkpoint is None:
        parser.error("demo requires --checkpoint PATH_TO_BEST_PT")
    chosen = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = torch.device(chosen if args.device == "auto" else args.device)
    if args.device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this Python environment.")
    args.amp = args.amp and args.device.type == "cuda"
    checkpoint = None
    if args.task == "demo":
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint["spec"]["task"] != "cifar10":
            parser.error("The Part 3.2 demo requires a CIFAR-10 checkpoint.")
        args.seed = checkpoint["seed"]  # Reproduce the original train/validation split.
    # Seed the random generators for repeatability. This does not guarantee
    # bit-for-bit identical results across hardware or CUDA implementations.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        # Let cuDNN choose fast convolution algorithms for these fixed image sizes.
        torch.backends.cudnn.benchmark = True
    args.output = args.output or (Path(__file__).resolve().parent / "part3_results" /
                                 f"{args.task}_{datetime.now():%Y%m%d_%H%M%S_%f}")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be new or empty to preserve earlier runs.")
    args.output.mkdir(parents=True, exist_ok=True)
    # Record settings and software/hardware details so results can be explained.
    config = {key: str(value) if isinstance(value, (Path, torch.device)) else value
              for key, value in vars(args).items()}
    config.update(python=platform.python_version(), torch=str(torch.__version__),
                  fused_sgd=args.device.type == "cuda",
                  gpu=torch.cuda.get_device_name(args.device) if args.device.type == "cuda" else None)
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("Device:", args.device, "GPU:", config["gpu"], "Mixed precision:", args.amp, flush=True)
    task = "cifar10" if args.task == "demo" else args.task
    (train_loader, val_loader, test_loader), spec = load_data(task, args)
    config.update(gpu_data_enabled=args.gpu_data and args.device.type == "cuda",
                  gpu_data_setup_seconds=args.gpu_data_setup_seconds)
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    model = make_model(spec).to(args.device)
    if args.device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    print(model)
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    # When AMP is disabled, the scaler leaves normal full-precision training unchanged.
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    if checkpoint is not None:
        if spec != checkpoint["spec"]:
            raise ValueError("Dataset metadata differs from the saved checkpoint.")
        model.load_state_dict(checkpoint["model_state"])
        sync(args.device)
        start = time.perf_counter()
        result = run_epoch(model, test_loader, args.device, args.amp, collect=True)
        sync(args.device)
        inference_seconds = time.perf_counter() - start
        save_evaluation(model, test_loader, spec, args, result)
        # Demonstrate one new epoch; the supplied checkpoint remains unchanged.
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr or 0.001, momentum=0.9,
                                    weight_decay=5e-4, fused=args.device.type == "cuda")
        sync(args.device)
        start = time.perf_counter()
        trained = run_epoch(model, train_loader, args.device, args.amp, optimizer, scaler)
        sync(args.device)
        summary = dict(mode="demo", test_accuracy=result["accuracy"],
                       full_test_inference_seconds=inference_seconds,
                       one_epoch_seconds=time.perf_counter() - start,
                       one_epoch_loss=trained["loss"], one_epoch_accuracy=trained["accuracy"],
                       host=platform.node(), gpu=config["gpu"],
                       note="Run this command on Rangpur during the demonstration; local execution is not cluster evidence.")
    else:
        epochs = args.epochs or 100
        # Linear learning-rate scaling is a heuristic: batch size 512 gives lr=0.4.
        # It does not guarantee identical optimisation to batch size 128 and lr=0.1.
        lr = args.lr or (0.1 * args.batch_size / 128)
        # Fused CUDA SGD also consumes AMP's overflow flag on-device.
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                    weight_decay=5e-4, fused=args.device.type == "cuda")
        # Momentum smooths SGD updates; weight decay penalises large weights.
        # The cosine schedule gradually lowers the learning rate across training.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        history, best_accuracy, best_epoch = [], -1.0, 0
        best_at_seconds = None
        validation_threshold_seconds = {"over_90_percent": None, "at_least_94_percent": None}
        sync(args.device)
        # Count cache construction/transfer in the training budget as well.
        started = time.perf_counter() - args.gpu_data_setup_seconds
        for epoch in range(1, epochs + 1):
            epoch_started = time.perf_counter()
            current_lr = optimizer.param_groups[0]["lr"]
            trained = run_epoch(model, train_loader, args.device, args.amp, optimizer, scaler)
            # Validate every 5 epochs by default to reduce overhead, and always at the end.
            # Only evaluated epochs can be selected as the best checkpoint.
            should_validate = epoch % args.val_every == 0 or epoch == epochs
            validated = run_epoch(model, val_loader, args.device, args.amp) if should_validate else None
            sync(args.device)
            elapsed = time.perf_counter() - started
            if validated is not None:
                for key, passed in (("over_90_percent", validated["accuracy"] > 0.90),
                                    ("at_least_94_percent", validated["accuracy"] >= 0.94)):
                    if passed and validation_threshold_seconds[key] is None:
                        validation_threshold_seconds[key] = elapsed
                # Select weights using validation accuracy only, never test accuracy.
                # This checkpoint stores model weights and metadata, not full optimiser state.
                if validated["accuracy"] > best_accuracy:
                    best_accuracy, best_epoch, best_at_seconds = validated["accuracy"], epoch, elapsed
                    torch.save(dict(model_state=model.state_dict(), spec=spec, seed=args.seed,
                                    epoch=epoch, validation_accuracy=best_accuracy,
                                    seconds_to_checkpoint=elapsed), args.output / "best.pt")
            history.append(dict(epoch=epoch, train_loss=trained["loss"], train_accuracy=trained["accuracy"],
                                val_loss=None if validated is None else validated["loss"],
                                val_accuracy=None if validated is None else validated["accuracy"],
                                lr=current_lr, epoch_seconds=time.perf_counter() - epoch_started,
                                elapsed_seconds=time.perf_counter() - started))
            scheduler.step()
            # Pending means validation was deliberately skipped on this epoch.
            val_text = "pending" if validated is None else f"{validated['accuracy']:.2%}"
            print(f"Epoch {epoch:3}/{epochs}: train={trained['accuracy']:.2%} "
                  f"val={val_text} loss={trained['loss']:.4f} "
                  f"elapsed={history[-1]['elapsed_seconds']:.1f}s", flush=True)
        sync(args.device)
        training_seconds = time.perf_counter() - started
        save_history(history, args.output)
        # Reload the best validation checkpoint, which may differ from the final epoch.
        # Evaluate the held-out test set after training to measure generalisation.
        selected = torch.load(args.output / "best.pt", map_location=args.device, weights_only=True)
        model.load_state_dict(selected["model_state"])
        result = run_epoch(model, test_loader, args.device, args.amp, collect=True)
        save_evaluation(model, test_loader, spec, args, result)
        summary = dict(task=task, test_accuracy=result["accuracy"], test_loss=result["loss"],
                       best_validation_accuracy=best_accuracy, selected_epoch=best_epoch,
                       seconds_to_selected_checkpoint=best_at_seconds,
                       total_training_seconds=training_seconds,
                       gpu_data_setup_seconds=args.gpu_data_setup_seconds,
                       validation_threshold_seconds=validation_threshold_seconds,
                       train_samples=len(train_loader.dataset), validation_samples=len(val_loader.dataset),
                       test_samples=len(test_loader.dataset), gpu=config["gpu"],
                       timing_definition="Training wall time includes GPU data-cache setup, validation and checkpoint saves; excludes downloads, model setup, final test and plots.",
                       evaluation="Best checkpoint selected only by validation accuracy; final test evaluated after training.")
        # Report measured target checks. Validation threshold times are not test results.
        if task == "cifar10":
            summary.update(over_90_percent_test=result["accuracy"] > 0.90,
                           at_least_94_percent_test=result["accuracy"] >= 0.94,
                           at_least_94_percent_and_total_training_under_360s=(
                               result["accuracy"] >= 0.94 and training_seconds <= 360),
                           note="Validation threshold times are not test accuracy benchmarks. Laptop results do not replace the Rangpur demonstration.")
    (args.output / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Results saved to:", args.output.resolve())


# Run only when launched as a script; this also supports multiprocessing on Windows.
if __name__ == "__main__":
    main()
