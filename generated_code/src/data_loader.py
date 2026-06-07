
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter
import numpy as np
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2

class HateMemeDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, mock_data=False, mock_size=100):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.mock_data = mock_data
        self.data = []

        if self.mock_data:
            self._generate_mock_data(mock_size)
        else:
            self._load_data()

        # Adversarial Augmentations (Image)
        self.aug_blur = A.GaussianBlur(p=1.0)
        self.aug_noise = A.GaussNoise(p=1.0)
        self.aug_color = A.ColorJitter(p=1.0)
        
    def _load_data(self):
        jsonl_path = os.path.join(self.data_dir, f'{self.split}.jsonl')
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")
            
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

    def _generate_mock_data(self, size):
        print(f"Generating {size} mock data samples for split '{self.split}'...")
        for i in range(size):
            self.data.append({
                "id": i,
                "img": f"mock_img_{i}.png",
                "text": f"This is a mock text sample {i} for testing purposes.",
                "label": random.randint(0, 1)
            })

    def _augment_image(self, image):
        # Randomly apply one of the adversarial perturbations
        aug_type = random.choice(['none', 'blur', 'noise', 'color'])
        if aug_type == 'none':
            return image
        
        img_np = np.array(image)
        if aug_type == 'blur':
            augmented = self.aug_blur(image=img_np)['image']
        elif aug_type == 'noise':
            augmented = self.aug_noise(image=img_np)['image']
        elif aug_type == 'color':
            augmented = self.aug_color(image=img_np)['image']
        
        return Image.fromarray(augmented)

    def _augment_text(self, text):
        # Simple adversarial text perturbations
        aug_type = random.choice(['none', 'char_swap', 'typo'])
        if aug_type == 'none':
            return text
        
        chars = list(text)
        if aug_type == 'char_swap' and len(chars) > 2:
            idx = random.randint(0, len(chars) - 2)
            chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
        elif aug_type == 'typo' and len(chars) > 0:
            idx = random.randint(0, len(chars) - 1)
            chars[idx] = chr(random.randint(97, 122)) # Random lowercase letter
            
        return "".join(chars)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['label']
        
        # Load Image
        if self.mock_data:
            # Generate a random colored image
            image = Image.new('RGB', (224, 224), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        else:
            img_path = os.path.join(self.data_dir, item['img'])
            try:
                image = Image.open(img_path).convert('RGB')
            except FileNotFoundError:
                # Fallback for missing images in real datset usually means corruption or path issue, 
                # but we'll return a blank one to not crash training
                print(f"Warning: Image {img_path} not found. Using black image.")
                image = Image.new('RGB', (224, 224), color=(0, 0, 0))

        # Apply Adversarial Augmentations (only during training)
        if self.split == 'train':
            image = self._augment_image(image)
            text = self._augment_text(text)

        if self.transform:
            image = self.transform(image)
            
        return {
            "id": item['id'],
            "image": image,
            "text": text,
            "label": torch.tensor(label, dtype=torch.long)
        }

if __name__ == "__main__":
    # Test Mock Data
    dataset = HateMemeDataset(data_dir=".", split='train', mock_data=True, mock_size=5)
    dataloader = DataLoader(dataset, batch_size=2)
    
    for batch in dataloader:
        print(f"Batch Text: {batch['text']}")
        print(f"Batch Label: {batch['label']}")
        break
