import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import warnings
import pandas as pd
from PIL import Image
from torchvision import transforms

from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings('ignore')

# Add the project root to sys.path so we can import src
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'generated_code'))
from src.feature_extractor import MultimodalFeatureExtractor

def evaluate_method_or_logic(X, y, description):
    print(f"Evaluating {description} with Image/Text OR Logic...")
    # Need at least one sample from each class
    if len(np.unique(y)) < 2:
        print(f"Skipping {description} due to only one class present in subset.")
        return 0.0, 0.0
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Identify split index
    mid = X.shape[1] // 2
    if X.shape[1] == 2816:
        mid = 2048 # ResNet embedding size is 2048, RoBERTa is 768
    elif X.shape[1] == 2817:
        mid = 2049 # ResNet + 1 Object Count = 2049, RoBERTa is 768
        
    X_train_img, X_train_text = X_train[:, :mid], X_train[:, mid:]
    X_test_img, X_test_text = X_test[:, :mid], X_test[:, mid:]
    
    # Image Classifier
    clf_img = MLPClassifier(hidden_layer_sizes=(128, 32), max_iter=200, random_state=42)
    clf_img.fit(X_train_img, y_train)
    
    # Text Classifier
    clf_text = MLPClassifier(hidden_layer_sizes=(128, 32), max_iter=200, random_state=43)
    clf_text.fit(X_train_text, y_train)
    
    # Individual Predictions
    y_pred_img = clf_img.predict(X_test_img)
    y_pred_text = clf_text.predict(X_test_text)
    
    # Logical OR Gate rule:
    # If image == 1 (Hateful) OR text == 1 (Hateful), then Hateful (1). Else Non-Hateful (0).
    y_pred = np.logical_or(y_pred_img == 1, y_pred_text == 1).astype(int)
    
    acc = accuracy_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred, average='macro') * 100
    
    print(f"[{description} (OR Gate)] Accuracy: {acc:.2f}%, Macro-F1: {f1:.2f}%")
    return acc, f1

def extract_subset_features(num_samples=150):
    print(f"Extracting raw multimodal features for {num_samples} samples from the dataset...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    base_dir = r"c:\Users\tarun\Desktop\Hima"
    data_dir = os.path.join(base_dir, "training", "TRAINING")
    csv_path = os.path.join(data_dir, "training.csv")
    
    df = pd.read_csv(csv_path, sep='\t')
    # Take a subset
    df = df.head(num_samples)
    
    file_names = df['file_name'].values
    labels = df['misogynous'].values
    texts = df['Text Transcription'].astype(str).values
    
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    print("Loading Object Detection Model...")
    object_detector = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    object_detector.to(device)
    object_detector.eval()
    
    extractor = MultimodalFeatureExtractor(device=device)
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    obj_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    batch_size = 32
    raw_features_list = []
    y_list = []
    
    for i in range(0, len(df), batch_size):
        end_idx = min(i + batch_size, len(df))
        batch_files = file_names[i:end_idx]
        batch_texts = texts[i:end_idx].tolist()
        batch_labels = labels[i:end_idx]
        
        batch_images = []
        batch_obj_images = []
        for j, fname in enumerate(batch_files):
            img_path = os.path.join(data_dir, fname)
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception:
                img = Image.new('RGB', (224, 224), color=(0,0,0))
            batch_images.append(img_transform(img))
            batch_obj_images.append(obj_transform(img).to(device))
            
        images_tensor = torch.stack(batch_images)
        features = extractor.extract_batch(images_tensor, batch_texts)
        
        # Object Detection features
        with torch.no_grad():
            obj_predictions = object_detector(batch_obj_images)
        
        # Count the number of high confidence objects
        obj_counts = []
        for pred in obj_predictions:
            count = (pred['scores'] > 0.5).sum().item()
            obj_counts.append([count])
            
        obj_counts = np.array(obj_counts)
        # Append the object count as an extra feature to the image portion of the multimodal features
        # Assuming features shape is (N, 2816) where :2048 is image. 
        # We append object counts so we get (N, 2817) where :2049 is image.
        enriched_features = np.hstack((features[:, :2048], obj_counts, features[:, 2048:]))
        
        raw_features_list.append(enriched_features)
        y_list.extend(batch_labels)
        print(f"Extracted {end_idx}/{num_samples} (with objects)...")
        
    X_raw = np.vstack(raw_features_list)
    y = np.array(y_list)
    return X_raw, y

def create_chart():
    # 1. Dynamically extract and evaluate the models based on the project's logic
    X_raw, y = extract_subset_features(num_samples=150)
    
    categories = ['Raw_Data', 'Incremental PCA', 'SMOTE', 'SMOTE_PCA']
    accuracy_scores = []
    macro_f1_scores = []
    
    # (1) Raw Data
    acc, f1 = evaluate_method_or_logic(X_raw, y, "Raw_Data")
    accuracy_scores.append(acc)
    macro_f1_scores.append(f1)
    
    # (2) Incremental PCA
    print("Applying Incremental PCA...")
    ipca = IncrementalPCA(n_components=30, batch_size=32)
    X_pca = ipca.fit_transform(X_raw)
    acc, f1 = evaluate_method_or_logic(X_pca, y, "Incremental PCA")
    accuracy_scores.append(acc)
    macro_f1_scores.append(f1)
    
    # (3) SMOTE
    print("Applying SMOTE on raw data...")
    smote = SMOTE(random_state=42, k_neighbors=min(3, len(y)//2))
    try:
        X_smote, y_smote = smote.fit_resample(X_raw, y)
        acc, f1 = evaluate_method_or_logic(X_smote, y_smote, "SMOTE")
    except Exception as e:
        print(f"SMOTE failed: {e}. Using raw results.")
        acc, f1 = accuracy_scores[0], macro_f1_scores[0]
    accuracy_scores.append(acc)
    macro_f1_scores.append(f1)
    
    # (4) SMOTE_PCA
    print("Applying SMOTE on PCA data...")
    try:
        X_smote_pca, y_smote_pca = smote.fit_resample(X_pca, y)
        acc, f1 = evaluate_method_or_logic(X_smote_pca, y_smote_pca, "SMOTE_PCA")
    except Exception as e:
        print(f"SMOTE_PCA failed: {e}. Using PCA results.")
        acc, f1 = accuracy_scores[1], macro_f1_scores[1]
    accuracy_scores.append(acc)
    macro_f1_scores.append(f1)

    # 2. Generate the chart
    x = np.arange(len(categories))  # the label locations
    width = 0.3  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 5))

    # Clean borders
    for spine in ax.spines.values():
        spine.set_color('#dddddd')

    rects1 = ax.bar(x - width/2 - 0.02, accuracy_scores, width, label='Accuracy', color='#1f77b4')
    rects2 = ax.bar(x + width/2 + 0.02, macro_f1_scores, width, label='Macro-F1', color='#d62728')

    # Formatting axis
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 110, 20))
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.tick_params(axis='both', which='both', length=0)
    ax.yaxis.grid(True, linestyle='-', which='major', color='#eeeeee', alpha=1.0)
    ax.set_axisbelow(True)

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, color='#333333')

    autolabel(rects1)
    autolabel(rects2)

    output_path = 'project_results_chart.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Chart successfully saved to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    create_chart()
