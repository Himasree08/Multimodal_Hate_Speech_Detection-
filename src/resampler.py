
import numpy as np
import os
from imblearn.over_sampling import SMOTE
from collections import Counter

class DataResampler:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.smote = SMOTE(random_state=42)

    def load_data(self):
        features_path = os.path.join(self.data_dir, 'features.npy')
        labels_path = os.path.join(self.data_dir, 'labels.npy')
        
        if not os.path.exists(features_path) or not os.path.exists(labels_path):
            raise FileNotFoundError("Features or labels not found. Run feature extraction first.")
            
        features = np.load(features_path)
        labels = np.load(labels_path)
        return features, labels

    def resample(self):
        print("Loading data...")
        X, y = self.load_data()
        print(f"Original class distribution: {Counter(y)}")
        
        print("Applying SMOTE...")
        if len(Counter(y)) < 2:
             print("Warning: Only 1 class detected. Skipping SMOTE.")
             X_resampled, y_resampled = X, y
        else:
             try:
                 X_resampled, y_resampled = self.smote.fit_resample(X, y)
                 print(f"Resampled class distribution: {Counter(y_resampled)}")
             except ValueError as e:
                 print(f"SMOTE failed (likely too few samples): {e}")
                 X_resampled, y_resampled = X, y
        
        save_X_path = os.path.join(self.data_dir, 'features_resampled.npy')
        save_y_path = os.path.join(self.data_dir, 'labels_resampled.npy')
        
        np.save(save_X_path, X_resampled)
        np.save(save_y_path, y_resampled)
        print(f"Saved resampled data to {self.data_dir}")

if __name__ == "__main__":
    # Test Run (requires features.npy to exist)
    try:
        resampler = DataResampler(data_dir="generated_code/data")
        resampler.resample()
    except Exception as e:
        print(f"Resampling failed (likely due to missing files in mock run): {e}")
