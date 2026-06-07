
import argparse
import os
from src.data_loader import HateMemeDataset
from src.feature_extractor import MultimodalFeatureExtractor
from src.resampler import DataResampler
from src.train_pipeline import TrainingPipeline
from torch.utils.data import DataLoader
from torchvision import transforms

def main():
    parser = argparse.ArgumentParser(description="Multimodal Hate Speech Detection")
    parser.add_argument("--data_dir", type=str, default="generated_code/data", help="Directory for data storage")
    parser.add_argument("--mock_data", action="store_true", help="Use mock data")
    parser.add_argument("--mock_size", type=int, default=100, help="Size of mock data")
    parser.add_argument("--step", type=str, choices=['all', 'extract', 'resample', 'train'], default='all', help="Step to run")
    args = parser.parse_args()

    # Ensure data directory exists
    if not os.path.exists(args.data_dir) and args.step != 'all' and not args.mock_data:
         # If not running all with mock data, we expect data dir to exist or specific sub-steps might fail
         pass 

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    # 1. Feature Extraction
    if args.step in ['all', 'extract']:
        print("Starting Feature Extraction...")
        dataset = HateMemeDataset(
            data_dir=args.data_dir, 
            split='train', 
            mock_data=args.mock_data, 
            mock_size=args.mock_size
        )
        # Add basic transform
        dataset.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Adjust n_components based on dataset size
        # Adjust n_components based on dataset size and batch size
        # IPCA requires n_components <= n_samples in a batch
        n_samples = len(dataset)
        batch_size = 32
        n_components = min(512, n_samples, batch_size)
        
        if args.mock_data:
             n_components = min(10, n_samples)
             
        print(f"Using n_components={n_components} for dataset size {n_samples} and batch size {batch_size}")
        
        extractor = MultimodalFeatureExtractor(n_components=n_components)
        
        extractor.fit_transform(dataloader, save_path=args.data_dir)
        print("Feature Extraction Complete.")

    # 2. Resampling (SMOTE)
    if args.step in ['all', 'resample']:
        print("Starting Resampling (SMOTE)...")
        resampler = DataResampler(data_dir=args.data_dir)
        resampler.resample()
        print("Resampling Complete.")

    # 3. Training
    if args.step in ['all', 'train']:
        print("Starting Training...")
        pipeline = TrainingPipeline(data_dir=args.data_dir)
        pipeline.train(epochs=50)
        print("Training Complete.")

if __name__ == "__main__":
    main()
