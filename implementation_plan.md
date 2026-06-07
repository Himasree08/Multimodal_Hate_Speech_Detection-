# Multimodal Hate Speech Detection Implementation Plan

This implementation plan outlines the steps to finalize, train, and deploy your Multimodal Hate Speech Detection system.

## 1. Project Overview
This project identifies hate speech in memes by combining visual and textual features.
*   **Architecture:** ResNet50 (Image) + RoBERTa (Text) → Incremental PCA → SMOTE (Oversampling) → MLP Classifier.
*   **Web App:** Flask-based interface generating predictions with confidence scores and heuristic checks (profanity/sentiment).

---

## 2. Prerequisites & Environment Setup
Ensure your environment is configured correctly.

**Recommended Python Version:** Python 3.8+

### 1. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 2. Install Dependencies
Create a file named `requirements.txt` in `c:\Users\tarun\Desktop\Hima\` with the following content:
```text
flask
torch
torchvision
transformers
scikit-learn
numpy
pandas
pillow
easyocr
textblob
better-profanity
joblib
tqdm
```

### 3. Install the libraries
```bash
pip install -r requirements.txt
```

---

## 3. Data Preparation
Your code expects data in specific locations.

1.  **JSONL Files:** Ensure `train.jsonl` and `dev.jsonl` are in `generated_code/data/`.
    *   *Current Status:* **Ready** (Files exist).
2.  **Images:** Place your training images in `generated_code/data/img/`.
    *   *Action Required:* Ensure the `img` folder contains the actual image files referenced in your `train.jsonl`.

---

## 4. Training Pipeline (How to Retrain)
The `main.py` script orchestrates the entire pipeline. Run these commands from the **root directory** (`c:\Users\tarun\Desktop\Hima\`).

### Option A: Run Everything (End-to-End)
```bash
python generated_code/main.py --step all
```

### Option B: Run Step-by-Step
1.  **Feature Extraction:** Extracts features using ResNet/RoBERTa and fits the PCA model.
    ```bash
    python generated_code/main.py --step extract
    ```
    *   *Output:* `features.npy`, `labels.npy`, `ipca.pkl` in `generated_code/data/`.

2.  **Resampling (SMOTE):** Balances variable classes in the dataset.
    ```bash
    python generated_code/main.py --step resample
    ```
    *   *Output:* `features_resampled.npy`, `labels_resampled.npy`.

3.  **Training:** Trains the MLP Classifier.
    ```bash
    python generated_code/main.py --step train
    ```
    *   *Output:* `model.pth` (Saved in the root directory).

---

## 5. Web Application Deployment
Once the model is trained, you can launch the Flask web app.

### 1. Run the App
```bash
python generated_code/app.py
```

### 2. Access the Interface
*   Open your browser and navigate to: `http://localhost:5000`

### 3. Test Integration
*   **Upload:** Upload a meme image (JPG/PNG).
*   **Text (Optional):** You can manually enter text. If left blank, **EasyOCR** will extract it from the image automatically.
*   **Result:** The app will display "Hateful" or "Non-Hateful" with a confidence score and explanation.

---

## 6. File Structure Reference
Ensure your files match this structure for the code to work without modification:

```text
Hima/
├── generated_code/
│   ├── app.py                # Flask Application
│   ├── main.py               # Training Entry Point
│   ├── data/
│   │   ├── train.jsonl       # Training Labels
│   │   ├── dev.jsonl         # Validation Labels
│   │   ├── ipca.pkl          # Generated PCA Model
│   │   ├── features.npy      # Generated Features
│   │   └── img/              # [IMPORTANT] Image storage
│   ├── src/
│   │   ├── classifier.py     # MLP Model Def
│   │   ├── data_loader.py    # Dataset Class
│   │   ├── feature_extractor.py 
│   │   ├── resampler.py      
│   │   └── train_pipeline.py 
│   └── templates/
│       └── index.html        # Web UI
├── model.pth                 # Trained Model (after training)
└── requirements.txt          # Dependencies
```

## 7. Next Steps for You
1.  **Check Images:** Verify `generated_code/data/img/` is not empty.
2.  **Install Requirements:** Run the pip install command.
3.  **Train:** Run `python generated_code/main.py --step all`.
4.  **Launch:** Run `python generated_code/app.py` and test with a few images.
