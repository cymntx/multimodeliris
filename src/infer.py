import os
import logging
from typing import List, Tuple
import yaml
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from model import MultiModalNet
from data import get_dataloaders

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def infer(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device
) -> Tuple[List[int], List[int]]:
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for fp, left, right, labels in loader:
            fp = fp.to(device, non_blocking=True)
            left = left.to(device, non_blocking=True)
            right = right.to(device, non_blocking=True)

            out = model(fp, left, right)
            preds = out.argmax(1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return all_preds, all_labels

def main():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    _, _, test_loader = get_dataloaders(
        cfg["data_path"], cfg["num_people"], cfg["batch_size"], cfg["num_workers"]
    )

    model = MultiModalNet(cfg["num_people"]).to(device)

    checkpoint_path = "checkpoints/best.pth"
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Checkpoint from epoch {checkpoint['epoch']} loaded")
    elif os.path.exists(cfg["model_path"]):
        logger.info(f"Loading model from {cfg['model_path']}")
        model.load_state_dict(torch.load(cfg["model_path"], map_location=device))
    else:
        logger.error("No trained model found. Please train the model first.")
        return

    logger.info("Running inference on test set...")
    predictions, ground_truth = infer(model, test_loader, device)

    accuracy = np.mean(np.array(predictions) == np.array(ground_truth))
    logger.info(f"Test Accuracy: {accuracy:.4f}")

    logger.info("\nClassification Report:")
    print(classification_report(ground_truth, predictions, zero_division=0))

    logger.info("\nConfusion Matrix:")
    cm = confusion_matrix(ground_truth, predictions)
    print(cm)

    logger.info("Inference complete.")

if __name__ == "__main__":
    main()