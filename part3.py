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
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset


  # Raw logits for cross entropy.


# Part 3.2: basic residual block, implemented directly (no pre-built models).
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
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
        for index, channels in enumerate((64, 128, 256, 512)):
            stages.extend([
                BasicBlock(in_channels, channels, stride=1 if index == 0 else 2),
                BasicBlock(channels, channels),
            ])
            in_channels = channels
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 10)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.fc(self.pool(self.stages(self.stem(x))).flatten(1))


def seed_worker(_):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def load_data(task, args):
    """Keep test data out of training, tuning, and checkpoint selection."""
    args.data_dir.mkdir(parents=True, exist_ok=True)
    from torchvision import datasets as vision_datasets, transforms
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
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
    val_full = vision_datasets.CIFAR10(
        str(args.data_dir), train=True, download=False, transform=eval_transform,
    )
    train_idx, val_idx = train_test_split(
        np.arange(len(train_full)), test_size=0.1,
        stratify=train_full.targets, random_state=args.seed,
    )
    test = vision_datasets.CIFAR10(
        str(args.data_dir), train=False, download=args.download, transform=eval_transform,
    )
    datasets = [Subset(train_full, train_idx), Subset(val_full, val_idx), test]
    spec = dict(task=task, classes=list(train_full.classes), mean=list(mean), std=list(std))
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
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_epoch(model, loader, device, amp, optimizer=None, scaler=None, collect=False):
    training = optimizer is not None
    model.train(training)
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), dtype=torch.long, device=device)
    total = 0
    truth, predictions = [], []
    criterion = nn.CrossEntropyLoss()
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            predicted = logits.detach().argmax(1)
            total += labels.size(0)
            total_loss += loss.detach().float() * labels.size(0)
            total_correct += (predicted == labels).sum()
            if collect:
                truth.extend(labels.cpu().tolist())
                predictions.extend(predicted.cpu().tolist())
    mean_loss = total_loss.item() / total
    if not math.isfinite(mean_loss):
        raise RuntimeError("Non-finite loss; try a lower learning rate or --no-amp.")
    return dict(loss=mean_loss, accuracy=total_correct.item() / total,
                truth=truth, predictions=predictions)


def save_history(history, output):
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
    labels = list(range(len(classes)))
    report = classification_report(result["truth"], result["predictions"],
                                   labels=labels, target_names=classes, zero_division=0)
    print(report)
    (args.output / "classification_report.txt").write_text(report, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(
        result["truth"], result["predictions"], labels=labels, display_labels=classes,
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
    shown = images.clone()
    if spec["task"] == "cifar10":
        shown = shown * torch.tensor(spec["std"])[None, :, None, None]
        shown += torch.tensor(spec["mean"])[None, :, None, None]
    fig, axes = plt.subplots(3, 4, figsize=(13, 8))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i >= len(shown):
            continue
        image = shown[i].permute(1, 2, 0).numpy().clip(0, 1)
        ax.imshow(image[..., 0] if image.shape[-1] == 1 else image, cmap="gray")
        ax.set_title(f"True: {classes[int(truth[i])]}\nPred: {classes[int(predicted[i])]}",
                     fontsize=9, color="green" if truth[i] == predicted[i] else "red")
    fig.tight_layout()
    fig.savefig(args.output / "predictions.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", choices=["cifar10", "demo"], nargs="?", default="cifar10")
    parser.add_argument("--checkpoint", type=Path, help="Saved best.pt; required for demo")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--output", type=Path, help="New/empty result directory")
    parser.add_argument("--epochs", type=int, help="Default: 100; demo always 1")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, help="Default: 0.1 for training, 0.001 for demo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0 if os.name == "nt" else 4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or (args.epochs is not None and args.epochs < 1):
        parser.error("Batch size/epochs must be positive and workers nonnegative.")
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    args.output = args.output or (Path(__file__).resolve().parent / "part3_results" /
                                 f"{args.task}_{datetime.now():%Y%m%d_%H%M%S_%f}")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be new or empty to preserve earlier runs.")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, (Path, torch.device)) else value
              for key, value in vars(args).items()}
    config.update(python=platform.python_version(), torch=str(torch.__version__),
                  gpu=torch.cuda.get_device_name(args.device) if args.device.type == "cuda" else None)
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("Device:", args.device, "GPU:", config["gpu"], "Mixed precision:", args.amp, flush=True)
    task = "cifar10" if args.task == "demo" else args.task
    (train_loader, val_loader, test_loader), spec = load_data(task, args)
    model = make_model(spec).to(args.device)
    if args.device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    print(model)
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
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
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr or 0.001, momentum=0.9, weight_decay=5e-4)
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
        lr = args.lr or 0.1
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        history, best_accuracy, best_epoch = [], -1.0, 0
        best_at_seconds = None
        validation_threshold_seconds = {"over_90_percent": None, "at_least_94_percent": None}
        sync(args.device)
        started = time.perf_counter()
        for epoch in range(1, epochs + 1):
            epoch_started = time.perf_counter()
            current_lr = optimizer.param_groups[0]["lr"]
            trained = run_epoch(model, train_loader, args.device, args.amp, optimizer, scaler)
            validated = run_epoch(model, val_loader, args.device, args.amp)
            sync(args.device)
            elapsed = time.perf_counter() - started
            for key, passed in (("over_90_percent", validated["accuracy"] > 0.90),
                                ("at_least_94_percent", validated["accuracy"] >= 0.94)):
                if passed and validation_threshold_seconds[key] is None:
                    validation_threshold_seconds[key] = elapsed
            if validated["accuracy"] > best_accuracy:
                best_accuracy, best_epoch, best_at_seconds = validated["accuracy"], epoch, elapsed
                torch.save(dict(model_state=model.state_dict(), spec=spec, seed=args.seed,
                                epoch=epoch, validation_accuracy=best_accuracy,
                                seconds_to_checkpoint=elapsed), args.output / "best.pt")
            history.append(dict(epoch=epoch, train_loss=trained["loss"], train_accuracy=trained["accuracy"],
                                val_loss=validated["loss"], val_accuracy=validated["accuracy"],
                                lr=current_lr, epoch_seconds=time.perf_counter() - epoch_started,
                                elapsed_seconds=time.perf_counter() - started))
            scheduler.step()
            print(f"Epoch {epoch:3}/{epochs}: train={trained['accuracy']:.2%} "
                  f"val={validated['accuracy']:.2%} loss={trained['loss']:.4f} "
                  f"elapsed={history[-1]['elapsed_seconds']:.1f}s", flush=True)
        sync(args.device)
        training_seconds = time.perf_counter() - started
        save_history(history, args.output)
        selected = torch.load(args.output / "best.pt", map_location=args.device, weights_only=True)
        model.load_state_dict(selected["model_state"])
        result = run_epoch(model, test_loader, args.device, args.amp, collect=True)
        save_evaluation(model, test_loader, spec, args, result)
        summary = dict(task=task, test_accuracy=result["accuracy"], test_loss=result["loss"],
                       best_validation_accuracy=best_accuracy, selected_epoch=best_epoch,
                       seconds_to_selected_checkpoint=best_at_seconds,
                       total_training_seconds=training_seconds,
                       validation_threshold_seconds=validation_threshold_seconds,
                       train_samples=len(train_loader.dataset), validation_samples=len(val_loader.dataset),
                       test_samples=len(test_loader.dataset), gpu=config["gpu"],
                       timing_definition="Training wall time includes validation and checkpoint saves; excludes downloads, model setup, final test and plots.",
                       evaluation="Best checkpoint selected only by validation accuracy; final test evaluated after training.")
        if task == "cifar10":
            summary.update(over_90_percent_test=result["accuracy"] > 0.90,
                           at_least_94_percent_test=result["accuracy"] >= 0.94,
                           at_least_94_percent_and_total_training_under_360s=(
                               result["accuracy"] >= 0.94 and training_seconds <= 360),
                           note="Validation threshold times are not test accuracy benchmarks. Laptop results do not replace the Rangpur demonstration.")
    (args.output / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Results saved to:", args.output.resolve())


if __name__ == "__main__":
    main()
