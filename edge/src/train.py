"""Training script for ECG classification model.

Resume support: this script saves a full checkpoint (model + optimizer +
scheduler + epoch + best score + history) to models/checkpoint.pth after
EVERY epoch. If you stop it (Ctrl+C, or just close the terminal) and rerun
later, it picks up exactly where it left off -- same epoch count, same
optimizer momentum, same "best so far" tracking, same loss/accuracy history.

models/best_model.pth remains just the clean weights of your best model
so far, for deployment / ONNX export -- it's never touched by the resume
logic itself, only updated when a new epoch actually beats the previous best.
"""

import os
import sys
import time
import json
import yaml
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tqdm import tqdm

from src.data_loader import load_config, create_dataloaders
from src.model import build_model, count_parameters

CHECKPOINT_PATH = 'models/checkpoint.pth'
BEST_MODEL_PATH = 'models/best_model.pth'
HISTORY_PATH = 'models/training_history.json'


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({'loss': loss.item(), 'acc': correct / total})

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, accuracy, macro_f1, np.array(all_preds), np.array(all_labels)


def save_checkpoint(path, model, optimizer, scheduler, next_epoch,
                     best_val_f1, patience_counter, history):
    """Full training state -- everything needed to resume exactly where
    you left off. Saved every epoch, overwriting the previous checkpoint."""
    torch.save({
        'next_epoch': next_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_f1': best_val_f1,
        'patience_counter': patience_counter,
        'history': history,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return (ckpt['next_epoch'], ckpt['best_val_f1'],
            ckpt['patience_counter'], ckpt['history'])


def train(config_path='configs/config.yaml'):
    # 1. Load config
    config = load_config(config_path)
    train_cfg = config.get('training', {})

    # 2. Set seed
    seed = train_cfg.get('seed', 42)
    set_seed(seed)

    # 3. Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 4. Create dataloaders
    train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(config)

    # 5. Build model
    model = build_model(config).to(device)
    print(f"Model parameters: {count_parameters(model)}")

    # 6. Criterion
    criterion = nn.CrossEntropyLoss()

    # 7. Optimizer
    optimizer = optim.Adam(model.parameters(),
                            lr=train_cfg.get('learning_rate', 0.001),
                            weight_decay=train_cfg.get('weight_decay', 0.0001))

    # 8. Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=train_cfg.get('scheduler_patience', 5),
        factor=0.5,
        mode='min'
    )

    os.makedirs('models', exist_ok=True)

    # 9. Resume from checkpoint if one exists -- this is the whole point.
    start_epoch = 0
    best_val_f1 = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Found checkpoint at {CHECKPOINT_PATH} -- resuming training.")
        start_epoch, best_val_f1, patience_counter, history = load_checkpoint(
            CHECKPOINT_PATH, model, optimizer, scheduler, device
        )
        print(f"Resuming at epoch {start_epoch + 1}, "
              f"best_val_f1 so far: {best_val_f1:.4f}, "
              f"patience_counter: {patience_counter}")
    else:
        print("No checkpoint found -- starting training from scratch.")

    early_stopping_patience = train_cfg.get('early_stopping_patience', 10)
    num_epochs = train_cfg.get('num_epochs', 50)

    # 10. Training loop
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

        lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Val F1: {val_f1:.4f} | LR: {lr}")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        scheduler.step(val_loss)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print("Saved new best model.")
            patience_counter = 0
        else:
            patience_counter += 1

        # Save the full checkpoint every epoch, regardless of whether this
        # epoch improved on the best score -- this is what lets you stop
        # today and resume tomorrow without losing progress.
        save_checkpoint(CHECKPOINT_PATH, model, optimizer, scheduler,
                         epoch + 1, best_val_f1, patience_counter, history)
        with open(HISTORY_PATH, 'w') as f:
            json.dump(history, f, indent=4)

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    # 11. After training (or after resuming and finishing)
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    test_loss, test_acc, test_f1, all_preds, all_labels = evaluate(model, test_loader, criterion, device)

    print("\nTest Evaluation:")
    print(classification_report(all_labels, all_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    print(f"Training complete. Best model saved to {BEST_MODEL_PATH}")
    print(f"To start a fresh run from scratch next time, delete {CHECKPOINT_PATH} first.")


if __name__ == '__main__':
    train()