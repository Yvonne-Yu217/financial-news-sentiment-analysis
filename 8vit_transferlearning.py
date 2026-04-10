import argparse
import csv
import datetime as dt
import json
import logging
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script 8: ViT transfer learning training with resumable artifacts"
    )
    parser.add_argument("--data-dir", default="training_dataset", help="Dataset root directory")
    parser.add_argument("--artifacts-dir", default="run_artifacts/8vit_transferlearning", help="Output artifacts directory")
    parser.add_argument("--years", nargs="+", type=int, default=None, help="Compatibility field with scripts 1-7")

    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--dropout-rate", type=float, default=0.25)
    parser.add_argument("--model-name", default="vit_base_patch16_224", help="timm model name")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")

    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")

    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--reprocess", action="store_true", help="Ignore existing final model and retrain")
    parser.add_argument("--save-plots", action="store_true", default=True)
    parser.add_argument("--no-save-plots", dest="save_plots", action="store_false")
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        default=False,
        help="Use class-weighted cross entropy based on training split",
    )
    parser.add_argument(
        "--no-class-weights",
        dest="use_class_weights",
        action="store_false",
        help="Disable class-weighted cross entropy",
    )

    return parser.parse_args()


def atomic_write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as exc:
            logging.error("Failed to load image %s: %s", image_path, exc)
            return self.__getitem__(0)


class ImprovedViTModel(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224", pretrained=True, num_classes=2, dropout_rate=0.2):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def remove_hidden_files(directory: Path):
    count = 0
    for fp in directory.rglob("._*"):
        if fp.is_file():
            fp.unlink()
            count += 1
    if count > 0:
        logging.info("Removed %d hidden files", count)


def detect_device(device_arg: str):
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_samples(data_dir: Path):
    classes = ["positive", "negative"]
    samples = []
    label_counts = {"positive": 0, "negative": 0}
    image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]

    for label, class_name in enumerate(classes):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            raise ValueError(f"Directory does not exist: {class_dir}")
        for pattern in image_patterns:
            for img in class_dir.glob(pattern):
                if img.name.startswith("._"):
                    continue
                samples.append((str(img), label))
                label_counts[class_name] += 1

    if not samples:
        raise ValueError(f"No images found in {data_dir}")

    logging.info("Found %d images", len(samples))
    logging.info("Positive images (0): %d", label_counts["positive"])
    logging.info("Negative images (1): %d", label_counts["negative"])
    return samples, label_counts


def make_splits(samples, train_ratio, val_ratio, seed):
    if not (0 < train_ratio < 1) or not (0 < val_ratio < 1) or train_ratio + val_ratio >= 1:
        raise ValueError("Invalid split ratios. Require 0<train_ratio<1, 0<val_ratio<1, train_ratio+val_ratio<1")

    indices = np.arange(len(samples))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    train_end = int(len(indices) * train_ratio)
    val_end = train_end + int(len(indices) * val_ratio)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]
    test_samples = [samples[i] for i in test_indices]
    return train_samples, val_samples, test_samples


def create_transforms():
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.2)),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        bar = tqdm(test_loader, desc="Testing")
        for inputs, labels in bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total_count += labels.size(0)
            total_correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            bar.set_postfix(loss=f"{total_loss / max(total_count, 1):.4f}", acc=f"{100.0 * total_correct / max(total_count, 1):.2f}%")

    accuracy = total_correct / max(total_count, 1)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1]).tolist()

    return {
        "test_loss": total_loss / max(total_count, 1),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


def write_test_result_tables(test_metrics, artifacts_dir: Path):
    """Write paper-ready test result tables in CSV and LaTeX formats."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    cm = test_metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    if len(cm) < 2 or len(cm[0]) < 2 or len(cm[1]) < 2:
        cm = [[0, 0], [0, 0]]

    tp = int(cm[0][0])
    fn = int(cm[0][1])
    fp = int(cm[1][0])
    tn = int(cm[1][1])

    cls_csv_path = artifacts_dir / "8vit_test_classification_results.csv"
    with cls_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Actual Class", "Predicted Positive", "Predicted Negative"])
        writer.writerow(["Positive", tp, fn])
        writer.writerow(["Negative", fp, tn])

    cls_tex_path = artifacts_dir / "8vit_test_classification_results.tex"
    cls_tex = "\n".join(
        [
            "\\begin{table}[h!]",
            "    \\caption{Test Set Classification Results}",
            "    \\label{tab:evaluation}",
            "    \\setlength{\\tabcolsep}{4pt}",
            "    \\small",
            "    \\begin{tabular}{lcc}",
            "        \\toprule",
            "        Actual Class & Predicted Positive & Predicted Negative \\\\",
            "        \\midrule",
            f"        Positive & {tp} & {fn} \\\\",
            f"        Negative & {fp} & {tn} \\\\",
            "        \\bottomrule",
            "    \\end{tabular}",
            "\\end{table}",
        ]
    )
    cls_tex_path.write_text(cls_tex + "\n", encoding="utf-8")

    metric_csv_path = artifacts_dir / "8vit_test_metrics.csv"
    with metric_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Accuracy", f"{test_metrics['accuracy'] * 100:.2f}%"])
        writer.writerow(["Precision", f"{test_metrics['precision'] * 100:.2f}%"])
        writer.writerow(["Recall", f"{test_metrics['recall'] * 100:.2f}%"])
        writer.writerow(["F1", f"{test_metrics['f1'] * 100:.2f}%"])
        writer.writerow(["Test Loss", f"{test_metrics['test_loss']:.4f}"])

    metric_tex_path = artifacts_dir / "8vit_test_metrics.tex"
    metric_tex = "\n".join(
        [
            "\\begin{table}[h!]",
            "    \\caption{ViT Test Metrics}",
            "    \\label{tab:vit_test_metrics}",
            "    \\setlength{\\tabcolsep}{6pt}",
            "    \\small",
            "    \\begin{tabular}{lc}",
            "        \\toprule",
            "        Metric & Value \\\\",
            "        \\midrule",
            f"        Accuracy & {test_metrics['accuracy'] * 100:.2f}\\% \\\\",
            f"        Precision & {test_metrics['precision'] * 100:.2f}\\% \\\\",
            f"        Recall & {test_metrics['recall'] * 100:.2f}\\% \\\\",
            f"        F1 & {test_metrics['f1'] * 100:.2f}\\% \\\\",
            f"        Test Loss & {test_metrics['test_loss']:.4f} \\\\",
            "        \\bottomrule",
            "    \\end{tabular}",
            "\\end{table}",
        ]
    )
    metric_tex_path.write_text(metric_tex + "\n", encoding="utf-8")

    return {
        "classification_csv": str(cls_csv_path),
        "classification_tex": str(cls_tex_path),
        "metrics_csv": str(metric_csv_path),
        "metrics_tex": str(metric_tex_path),
    }


def save_plot(values_a, values_b, label_a, label_b, ylabel, title, out_path: Path):
    plt.figure(figsize=(10, 6))
    plt.plot(values_a, label=label_a)
    plt.plot(values_b, label=label_b)
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    args = parse_args()
    set_seed(args.seed)

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_best_path = artifacts_dir / "8vit_best_model.pth"
    model_final_path = artifacts_dir / "8vit_final_model.pth"
    checkpoint_path = artifacts_dir / "8vit_transferlearning_checkpoint.pt"
    status_path = artifacts_dir / "8vit_transferlearning_status.json"
    summary_path = artifacts_dir / "8vit_transferlearning_summary.json"
    metrics_path = artifacts_dir / "8vit_training_metrics.json"

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    status = {
        "script": "8vit_transferlearning.py",
        "run_id": run_id,
        "started_at": dt.datetime.now().isoformat(),
        "last_updated": dt.datetime.now().isoformat(),
        "phase": "initializing",
        "completed": False,
        "current_epoch": 0,
        "total_epochs": args.num_epochs,
        "config": vars(args),
    }
    atomic_write_json(status_path, status)

    if model_final_path.exists() and not args.reprocess and not args.resume:
        status.update(
            {
                "phase": "completed",
                "completed": True,
                "last_updated": dt.datetime.now().isoformat(),
                "message": "final model exists; skipped (use --reprocess to force retrain)",
            }
        )
        atomic_write_json(status_path, status)

        summary = {
            "script": "8vit_transferlearning.py",
            "run_id": run_id,
            "status": "skipped",
            "reason": "final model exists",
            "artifacts_dir": str(artifacts_dir),
            "final_model_path": str(model_final_path),
            "config": vars(args),
            "finished_at": dt.datetime.now().isoformat(),
        }
        atomic_write_json(summary_path, summary)
        logging.info("Final model already exists at %s; skipped.", model_final_path)
        return

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    remove_hidden_files(data_dir)

    status.update({"phase": "loading_data", "last_updated": dt.datetime.now().isoformat()})
    atomic_write_json(status_path, status)

    samples, label_counts = load_samples(data_dir)
    train_samples, val_samples, test_samples = make_splits(samples, args.train_ratio, args.val_ratio, args.seed)

    train_transform, eval_transform = create_transforms()

    train_dataset = ImageDataset(train_samples, transform=train_transform)
    val_dataset = ImageDataset(val_samples, transform=eval_transform)
    test_dataset = ImageDataset(test_samples, transform=eval_transform)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)

    device = detect_device(args.device)
    logging.info("Using device: %s", device)
    logging.info(
        "Dataset split: train=%d, val=%d, test=%d",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )

    model = ImprovedViTModel(
        model_name=args.model_name,
        pretrained=args.pretrained,
        dropout_rate=args.dropout_rate,
    ).to(device)

    class_weights = None
    if args.use_class_weights:
        train_label_counts = np.bincount([label for _, label in train_samples], minlength=2)
        total_train = max(int(train_label_counts.sum()), 1)
        # Inverse-frequency weighting to reduce bias toward majority class.
        class_weights = torch.tensor(
            [total_train / max(int(c), 1) for c in train_label_counts],
            dtype=torch.float32,
            device=device,
        )
        class_weights = class_weights / class_weights.mean()
        logging.info(
            "Using class weights: counts=%s weights=%s",
            train_label_counts.tolist(),
            [round(float(x), 4) for x in class_weights.detach().cpu().tolist()],
        )

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    start_epoch = 0
    best_val_acc = 0.0
    no_improve_epochs = 0
    history = {
        "train_losses": [],
        "val_losses": [],
        "train_accuracies": [],
        "val_accuracies": [],
    }

    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint["best_val_acc"]
        no_improve_epochs = checkpoint["no_improve_epochs"]
        history = checkpoint["history"]
        logging.info("Resumed from epoch %d", start_epoch)

    status.update({"phase": "training", "last_updated": dt.datetime.now().isoformat()})
    atomic_write_json(status_path, status)

    total_start = time.time()

    for epoch in range(start_epoch, args.num_epochs):
        epoch_start = time.time()
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(train_loader, desc=f"Training {epoch + 1}/{args.num_epochs}")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            train_bar.set_postfix(loss=f"{train_loss / max(train_total, 1):.4f}", acc=f"{100.0 * train_correct / max(train_total, 1):.2f}%")

        epoch_train_loss = train_loss / max(train_total, 1)
        epoch_train_acc = train_correct / max(train_total, 1)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Validation {epoch + 1}/{args.num_epochs}")
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                val_bar.set_postfix(loss=f"{val_loss / max(val_total, 1):.4f}", acc=f"{100.0 * val_correct / max(val_total, 1):.2f}%")

        epoch_val_loss = val_loss / max(val_total, 1)
        epoch_val_acc = val_correct / max(val_total, 1)
        scheduler.step(epoch_val_loss)

        history["train_losses"].append(epoch_train_loss)
        history["val_losses"].append(epoch_val_loss)
        history["train_accuracies"].append(epoch_train_acc)
        history["val_accuracies"].append(epoch_val_acc)

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            no_improve_epochs = 0
            torch.save(model.state_dict(), model_best_path)
        else:
            no_improve_epochs += 1

        status.update(
            {
                "last_updated": dt.datetime.now().isoformat(),
                "current_epoch": epoch + 1,
                "best_val_acc": best_val_acc,
                "epoch_train_loss": epoch_train_loss,
                "epoch_val_loss": epoch_val_loss,
                "epoch_train_acc": epoch_train_acc,
                "epoch_val_acc": epoch_val_acc,
            }
        )
        atomic_write_json(status_path, status)

        if (epoch + 1) % max(args.checkpoint_every, 1) == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_acc": best_val_acc,
                    "no_improve_epochs": no_improve_epochs,
                    "history": history,
                    "config": vars(args),
                },
                checkpoint_path,
            )

        epoch_time = int(time.time() - epoch_start)
        logging.info(
            "Epoch %d/%d | train_acc=%.2f%% val_acc=%.2f%% | duration=%s",
            epoch + 1,
            args.num_epochs,
            epoch_train_acc * 100,
            epoch_val_acc * 100,
            str(dt.timedelta(seconds=epoch_time)),
        )

        if no_improve_epochs >= args.patience:
            logging.info("Early stopping triggered after %d epochs without improvement", args.patience)
            break

    if not model_best_path.exists():
        torch.save(model.state_dict(), model_best_path)

    model.load_state_dict(torch.load(model_best_path, map_location=device))
    test_metrics = evaluate_model(model, test_loader, criterion, device)

    torch.save(model.state_dict(), model_final_path)

    metrics_payload = {
        "history": history,
        "test": test_metrics,
    }
    atomic_write_json(metrics_path, metrics_payload)

    table_artifacts = write_test_result_tables(test_metrics, artifacts_dir)

    if args.save_plots:
        save_plot(
            history["train_accuracies"],
            history["val_accuracies"],
            "Training Accuracy",
            "Validation Accuracy",
            "Accuracy",
            "Training and Validation Accuracy",
            artifacts_dir / "8vit_training_accuracy.png",
        )
        save_plot(
            history["train_losses"],
            history["val_losses"],
            "Training Loss",
            "Validation Loss",
            "Loss",
            "Training and Validation Loss",
            artifacts_dir / "8vit_training_loss.png",
        )

    status.update(
        {
            "phase": "completed",
            "completed": True,
            "last_updated": dt.datetime.now().isoformat(),
        }
    )
    atomic_write_json(status_path, status)

    summary = {
        "script": "8vit_transferlearning.py",
        "run_id": run_id,
        "status": "completed",
        "started_at": status["started_at"],
        "finished_at": dt.datetime.now().isoformat(),
        "duration_seconds": int(time.time() - total_start),
        "device": str(device),
        "dataset": {
            "data_dir": str(data_dir),
            "total_samples": len(samples),
            "label_counts": label_counts,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
        },
        "training": {
            "use_class_weights": args.use_class_weights,
            "class_weights": [float(x) for x in class_weights.detach().cpu().tolist()] if class_weights is not None else None,
        },
        "best_val_acc": best_val_acc,
        "test_metrics": test_metrics,
        "artifacts": {
            "best_model": str(model_best_path),
            "final_model": str(model_final_path),
            "checkpoint": str(checkpoint_path),
            "status": str(status_path),
            "metrics": str(metrics_path),
            "test_tables": table_artifacts,
        },
        "config": vars(args),
    }
    atomic_write_json(summary_path, summary)

    logging.info("Training completed. Summary: %s", summary_path)
    logging.info("Test accuracy: %.2f%%", test_metrics["accuracy"] * 100)


if __name__ == "__main__":
    main()
