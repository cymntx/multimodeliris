import os
import logging
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data import get_dataloaders
from model import MultiModalNet

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0, 0
    for batch in loader:
        fp, left, right, labels = [x.to(device) for x in batch]
        optimizer.zero_grad()
        outputs = model(fp, left, right)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * fp.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for batch in loader:
            fp, left, right, labels = [x.to(device) for x in batch]
            outputs = model(fp, left, right)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * fp.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def train(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = get_dataloaders(
        config["data_path"],
        config["num_people"],
        config["batch_size"],
        config["num_workers"],
        config["train_split"],
        config["val_split"],
        config["augment_train"],
    )
    model = MultiModalNet(config["num_people"], config["dropout"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"])
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=config["patience"], verbose=True
    )
    best_val_loss = float("inf")
    for epoch in range(config["epochs"]):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        logger.info(
            f"Epoch {epoch + 1}/{config['epochs']} - Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config["model_path"])
            logger.info(f"Saved best model to {config['model_path']}")
if __name__ == "__main__":
    train("src/config.yaml")
