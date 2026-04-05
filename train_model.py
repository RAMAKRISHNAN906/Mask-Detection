"""
Face Mask Detection - Model Training Script (PyTorch)
Uses MobileNetV2 transfer learning for mask/no-mask classification.

Dataset Structure:
  dataset/
    with_mask/       -> images of people wearing masks
    without_mask/    -> images of people without masks

Download dataset from:
  https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset
  or
  https://github.com/prajnasb/observations/tree/master/experiements/data
"""

import os
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# ── Configuration ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "mask_detector.pth")
PLOT_SAVE_PATH = os.path.join(BASE_DIR, "static", "images", "training_plot.png")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data_transforms():
    return {
        "train": transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        "val": transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }


def build_model(num_classes=2):
    """Build MobileNetV2-based mask detection model."""
    print("[INFO] Building MobileNetV2 model...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze feature extractor
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.last_channel, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes),
    )

    return model.to(DEVICE)


def train():
    """Full training pipeline."""
    if not os.path.exists(DATASET_DIR):
        print(f"[ERROR] Dataset directory not found: {DATASET_DIR}")
        print("  Please create the following structure:")
        print("    dataset/with_mask/    -> mask images")
        print("    dataset/without_mask/ -> no-mask images")
        print("\n  Download from: https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset")
        return

    subdirs = [d for d in os.listdir(DATASET_DIR)
               if os.path.isdir(os.path.join(DATASET_DIR, d))]
    if len(subdirs) < 2:
        print("[ERROR] Need at least 2 class folders in dataset/")
        print(f"  Found: {subdirs}")
        print("  Expected: with_mask, without_mask")
        return

    data_transforms = get_data_transforms()

    full_dataset = datasets.ImageFolder(DATASET_DIR, transform=data_transforms["train"])
    class_names = full_dataset.classes
    print(f"[INFO] Classes found: {class_names}")
    print(f"[INFO] Total images: {len(full_dataset)}")

    # Split 80/20
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print(f"  Device: {DEVICE}")

    model = build_model(num_classes=len(class_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\n[INFO] Training for {EPOCHS} epochs...\n")

    for epoch in range(EPOCHS):
        start = time.time()

        # Train
        model.train()
        running_loss, running_corrects, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += inputs.size(0)

        train_loss = running_loss / total
        train_acc = running_corrects / total

        # Validate
        model.eval()
        val_loss_sum, val_corrects, val_total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss_sum += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels).item()
                val_total += inputs.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_corrects / val_total

        scheduler.step()
        elapsed = time.time() - start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  Epoch {epoch + 1}/{EPOCHS} ({elapsed:.1f}s) - "
              f"Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    # Save best model
    model.load_state_dict(best_model_wts)
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "img_size": IMG_SIZE,
    }, MODEL_SAVE_PATH)

    print(f"\n[INFO] Best Val Accuracy: {best_acc:.4f}")
    print(f"[INFO] Model saved to {MODEL_SAVE_PATH}")

    # Classification report
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n" + classification_report(all_labels, all_preds, target_names=class_names))

    # Save training plot
    plt.style.use("ggplot")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(history["train_acc"], label="Train Accuracy")
    ax2.plot(history["val_acc"], label="Val Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(PLOT_SAVE_PATH), exist_ok=True)
    plt.savefig(PLOT_SAVE_PATH, dpi=150)
    print(f"[INFO] Training plot saved to {PLOT_SAVE_PATH}")

    print("\n[SUCCESS] Training complete!")


if __name__ == "__main__":
    train()
