
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import RobertaModel, RobertaTokenizer
from torch.utils.data import DataLoader
from sklearn.decomposition import IncrementalPCA
import numpy as np
from tqdm import tqdm
import joblib
from data_loader import HateMemeDataset

class MultimodalFeatureExtractor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', n_components=512):
        self.device = device
        self.n_components = n_components
        
        # Image Model (ResNet50)
        self.resnet = models.resnet50(weights='DEFAULT')
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1]) # Remove classification layer
        self.resnet.to(self.device)
        self.resnet.eval()
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Text Model (RoBERTa)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.roberta.to(self.device)
        self.roberta.eval()
        
        # PCA
        self.ipca = IncrementalPCA(n_components=self.n_components, batch_size=200)

    def extract_batch(self, images, texts):
        # Image Features
        images = images.to(self.device)
        with torch.no_grad():
            img_features = self.resnet(images)
            img_features = img_features.view(img_features.size(0), -1) # (B, 2048)
            
        # Text Features
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_outputs = self.roberta(**inputs)
            text_features = text_outputs.last_hidden_state[:, 0, :] # CLS token (B, 768)
            
        # Concatenate
        combined_features = torch.cat((img_features, text_features), dim=1) # (B, 2816)
        return combined_features.cpu().numpy()

    def fit_transform(self, dataloader, save_path):
        all_features = []
        all_labels = []
        
        print("Extracting features and fitting IPCA...")
        for batch in tqdm(dataloader):
            images = batch['image']
            texts = batch['text']
            labels = batch['label']
            
            features = self.extract_batch(images, texts)
            self.ipca.partial_fit(features)
            
        print("Transforming features...")
        # Reset dataloader or iterate again if needed, but for IPCA we usually fit then transform.
        # However, to save memory we might want to transform in batches too.
        # Since IPCA is fitted, we can now transform.
        
        for batch in tqdm(dataloader):
            images = batch['image']
            texts = batch['text']
            labels = batch['label']
            
            features = self.extract_batch(images, texts)
            reduced_features = self.ipca.transform(features)
            
            all_features.append(reduced_features)
            all_labels.extend(labels.numpy())
            
        all_features = np.vstack(all_features)
        all_labels = np.array(all_labels)
        
        np.save(os.path.join(save_path, 'features.npy'), all_features)
        np.save(os.path.join(save_path, 'labels.npy'), all_labels)
        
        # Save IPCA model
        joblib.dump(self.ipca, os.path.join(save_path, 'ipca.pkl'))
        print(f"Saved IPCA model to {os.path.join(save_path, 'ipca.pkl')}")
        print(f"Saved extracted features with shape {all_features.shape}")

if __name__ == "__main__":
    # Mock Run
    dataset = HateMemeDataset(data_dir=".", split='train', mock_data=True, mock_size=50)
    dataset.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False) # Shuffle false to keep order for transform if needed (but we do batch-wise)
    
    extractor = MultimodalFeatureExtractor()
    extractor.fit_transform(dataloader, save_path="generated_code/data")
