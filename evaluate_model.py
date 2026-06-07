import os
import sys
import torch
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

# ✅ Fix path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "generated_code", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ✅ Manual config
DATA_DIR = "generated_code/data"
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
TRAIN_FILE = "train.jsonl"

# ✅ Correct imports
try:
    from data_loader import HateMemeDataset
    from feature_extractor import MultimodalFeatureExtractor
    from classifier import MLPClassifier
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current path: {sys.path}")
    print(f"SRC_DIR: {SRC_DIR}")
    print(f"SRC_DIR exists: {os.path.exists(SRC_DIR)}")
    raise


def evaluate_model():
    print("Starting Evaluation...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])

    dataset = HateMemeDataset(TRAIN_FILE, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    # Load PCA
    with open("generated_code/data/ipca.pkl", "rb") as f:
        pca = pickle.load(f)

    extractor = MultimodalFeatureExtractor(device=device)

    model = SimpleMLP(input_dim=pca.n_components_, num_classes=2).to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    preds, targets = [], []

    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch['image']
            texts = batch['text']
            labels = batch['label']

            features = extractor(images, texts)

            print("Before PCA:", features.shape)
            features = pca.transform(features)
            print("After PCA:", features.shape)

            inputs = torch.tensor(features, dtype=torch.float32).to(device)
            outputs = model(inputs)

            pred = torch.argmax(outputs, dim=1).cpu().numpy()

            preds.extend(pred)
            targets.extend(labels.numpy())

    print("\nAccuracy:", accuracy_score(targets, preds))
    print("F1 Score:", f1_score(targets, preds))

    print("\nReport:\n", classification_report(targets, preds))

    cm = confusion_matrix(targets, preds)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.savefig("confusion_matrix.png")


if __name__ == "__main__":
    evaluate_model()