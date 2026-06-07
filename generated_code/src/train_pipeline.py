
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from src.classifier import MLPClassifier, get_optimizer, get_criterion
from sklearn.metrics import accuracy_score, f1_score

class TrainingPipeline:
    def __init__(self, data_dir, model_save_path='model.pth'):
        self.data_dir = data_dir
        self.model_save_path = model_save_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def load_resampled_data(self):
        features_path = os.path.join(self.data_dir, 'features_resampled.npy')
        labels_path = os.path.join(self.data_dir, 'labels_resampled.npy')
        
        if not os.path.exists(features_path):
            print("Resampled data not found, falling back to original features...")
            features_path = os.path.join(self.data_dir, 'features.npy')
            labels_path = os.path.join(self.data_dir, 'labels.npy')
            
        X = np.load(features_path)
        y = np.load(labels_path)
        
        # Convert to Tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=32, shuffle=True)

    def train(self, epochs=10):
        dataloader = self.load_resampled_data()
        
        # Determine input dimension from data
        sample_batch, _ = next(iter(dataloader))
        input_dim = sample_batch.shape[1]
        print(f"Detected input dimension: {input_dim}")
        
        model = MLPClassifier(input_dim=input_dim).to(self.device)
        optimizer = get_optimizer(model)
        criterion = get_criterion()
        
        print(f"Starting training on {self.device}...")
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            all_preds = []
            all_labels = []
            
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
                
            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='macro')
            
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")
            
        torch.save(model.state_dict(), self.model_save_path)
        print(f"Model saved to {self.model_save_path}")

if __name__ == "__main__":
    pipeline = TrainingPipeline(data_dir="generated_code/data")
    pipeline.train(epochs=50)
