import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from src.models.models import LSTMClassifier, TransEncClassifier
from src.data.dataset import create_dataset
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from tqdm import tqdm


def train(model, train_dataloader, device, optimizer, criterion, epoch, EPOCHS):
    model.train()
    model.to(device)
    train_progress = tqdm(train_dataloader, colour="cyan")
    running_loss = 0.0
    for idx, (xb, yb) in enumerate(train_progress):
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        optimizer.zero_grad()
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_progress.set_description(
            f"TRAIN | Epoch: {epoch}/{EPOCHS} | Iter: {idx}/{len(train_dataloader)} | \
                Loss: {loss.item():.4f}"
        )
    return running_loss / len(train_dataloader)


def evaluate(model, val_dataloader, device, criterion):
    model.eval()
    all_loss = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for idx, (xb, yb) in enumerate(val_dataloader):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            all_loss.append(loss.cpu().item())
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(yb.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    loss = np.mean(all_loss)
    f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"VAL | Loss: {loss:.4f} | F1: {f1:.4f}")
    return loss, f1


def plot_metrics(train_losses, val_losses, val_ious):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(val_ious, label="Val f1")
    plt.xlabel("Epoch")
    plt.ylabel("f1")
    plt.title("Validation f1 over Epochs")
    plt.legend()
    plt.tight_layout()
    # plt.savefig("training_metrics.png")
    # plt.close()


def train_seq_model(
    model,
    train_ds,
    val_ds,
    epochs=30,
    lr=3e-4,
    batch_size=16,
    patience=20,
    MODEL_SAVE_PATH="experiments/weights",
):
    if not os.path.isdir(MODEL_SAVE_PATH):
        os.mkdir(MODEL_SAVE_PATH)
    train_loader, val_loader = (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    best_predict = -1
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    val_f1 = []

    for epoch in range(epochs):
        train_loss = train(
            model, train_loader, device, optimizer, criterion, epoch, epochs
        )
        val_loss, f1 = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1.append(f1)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "val_f1": f1,
        }
        torch.save(checkpoint, os.path.join(MODEL_SAVE_PATH, "last.pt"))

        if f1 > best_predict:
            torch.save(checkpoint, os.path.join(MODEL_SAVE_PATH, "best.pt"))
            best_predict = f1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve > patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    plot_metrics(train_losses, val_losses, val_f1)


if __name__ == "__main__":
    train_ds, val_ds, test_ds = create_dataset(
        processed=True,
        train_input_dir="data/ts_train/npy",
        ts_data="data/drought_indices.csv",
        scaling_dir="data/ts_train/scaler",
    )

    model = TransEncClassifier(feat_dim=6)
    train_seq_model(
        model, train_ds, val_ds, epochs=30, lr=3e-4, batch_size=16, patience=30
    )
