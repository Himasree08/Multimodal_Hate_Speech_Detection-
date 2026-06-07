# Multimodal_Hate_Speech_Detection-
Effective Multimodal Hate Speech Detection Using ResNet50 and RoBERTa
This project detects hateful content in memes by analyzing both image and text information using deep learning and machine learning techniques.

## Problem Statement

Traditional hate speech detection systems focus only on text. Memes combine image and text, making detection more challenging. This project addresses the problem through multimodal learning.

## Technologies Used

- Python
- Flask
- PyTorch
- ResNet50
- RoBERTa
- EasyOCR
- Scikit-learn
- Incremental PCA
- SMOTE
- NumPy
- Pandas

## System Workflow

1. Upload Meme Image
2. Extract Text using EasyOCR
3. Extract Image Features using ResNet50
4. Extract Text Features using RoBERTa
5. Combine Features
6. Apply Incremental PCA
7. Apply SMOTE
8. Predict using MLP Classifier
9. Display Result

## Results

- Accuracy: 85% – 88%
- Macro F1 Score: 83% – 87%

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Open:

```text
http://localhost:5000
```

## Future Enhancements

- Cloud Deployment
- Multilingual Support
- Advanced Transformer Models

## Contributors

- G Hima Sree (Team Lead)
- Team Members
