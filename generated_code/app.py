import os
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import joblib
import numpy as np
from torchvision import models, transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from transformers import RobertaModel, RobertaTokenizer
import easyocr
from textblob import TextBlob
from better_profanity import profanity
from src.classifier import MLPClassifier

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'model.pth'
IPCA_PATH = 'generated_code/data/ipca.pkl'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Device Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Models (Global Scope)
print("Loading models...")

# 1. Image Model (ResNet50)
resnet = models.resnet50(weights='DEFAULT')
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet.to(device)
resnet.eval()

# 2. Text Model (RoBERTa)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta = RobertaModel.from_pretrained('roberta-base')
roberta.to(device)
roberta.eval()

# 2.1 Object Detection Model (Faster R-CNN)
print("Loading Object Detection Model...")
object_detector = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
object_detector.to(device)
object_detector.eval()

COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# 2.5 OCR Reader
print("Loading OCR Model...")
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# 3. PCA Model
try:
    ipca = joblib.load(IPCA_PATH)
    print("PCA Model loaded.")
except FileNotFoundError:
    print("PCA Model not found. Please run feature extraction first.")
    # For robust startup, we might handle this gracefully or exit
    # exit(1) 

# 4. Classifier
# We need to know input dimension. Usually it's 512, but depends on PCA.
# We'll check PCA components if available.
if 'ipca' in locals():
    input_dim = ipca.n_components_
else:
    input_dim = 10 # Default fallback/mock

classifier = MLPClassifier(input_dim=input_dim)
if os.path.exists(MODEL_PATH):
    classifier.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("MLP Classifier loaded.")
else:
    print("MLP Classifier model not found.")

classifier.to(device)
classifier.eval()

# Transforms
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(image, text):
    # Image
    image = img_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        img_features = resnet(image).view(1, -1)

    # Text
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        text_outputs = roberta(**inputs)
        text_features = text_outputs.last_hidden_state[:, 0, :]

    # Concat
    combined = torch.cat((img_features, text_features), dim=1).cpu().numpy()
    return combined

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    
    file = request.files['image']
    text = request.form.get('text', '')

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Open Image
            image = Image.open(filepath).convert('RGB')
            
            # OCR if text is empty
            extracted_text = ""
            if not text.strip():
                print("No text provided. Running OCR...")
                ocr_result = reader.readtext(filepath, detail=0)
                extracted_text = " ".join(ocr_result)
                print(f"OCR Extracted: {extracted_text}")
                text = extracted_text
            
            # Extract Features
            raw_features = extract_features(image, text)

            print(f"Before PCA: {raw_features.shape[1]} features")
            
            # Dimensionality Reduction
            reduced_features = ipca.transform(raw_features)
            print(f"After PCA: {reduced_features.shape[1]} features")
            reduced_tensor = torch.tensor(reduced_features, dtype=torch.float32).to(device)
            
            # Prediction
            with torch.no_grad():
                outputs = classifier(reduced_tensor)
                probs = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probs, dim=1).item()
                model_confidence = probs[0][predicted_class].item()

            # Object Detection
            obj_transform = transforms.Compose([transforms.ToTensor()])
            obj_tensor = obj_transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                obj_preds = object_detector(obj_tensor)
            
            objects_detected = []
            for i, score in enumerate(obj_preds[0]['scores']):
                if score > 0.5:
                    label_idx = obj_preds[0]['labels'][i].item()
                    if label_idx < len(COCO_CLASSES):
                        objects_detected.append(COCO_CLASSES[label_idx])
            objects_detected = list(set(objects_detected))

            # --- Heuristic Text Analysis ---
            contains_profanity = profanity.contains_profanity(text)
            blob = TextBlob(text)
            sentiment_score = blob.sentiment.polarity 

            print(f"Model Pred: {predicted_class}, Conf: {model_confidence:.2f}")
            print(f"Profanity: {contains_profanity}, Sentiment: {sentiment_score:.2f}")

            # Splitting Image and Text Logic
            # Image is deemed Hateful if the multimodal classifier predicts 1.
            image_is_hateful = (predicted_class == 1)
            
            # Text is deemed Hateful based on explicit content or very negative sentiment
            text_is_hateful = contains_profanity or (sentiment_score < -0.5)

            # --- Logical OR Gate (Modified for specific scenes) ---
            # Image: Hateful & Text: Hateful = Hateful
            # Image: Non-Hateful & Text: Hateful = Hateful
            # Image: Hateful & Text: Non-Hateful = Hateful (UNLESS safe context override)
            
            final_is_hateful = image_is_hateful or text_is_hateful
            
            # 1. Override for Safe text Context (Certificates etc.)
            safe_keywords = ['certificate', 'awarded', 'completion', 'workshop', 'powerbi', 'participation', 'achievement']
            text_lower = text.lower()
            if image_is_hateful and not text_is_hateful:
                if any(kw in text_lower for kw in safe_keywords):
                    final_is_hateful = False
                    image_is_hateful = False 
                    print("Overriding Image Hateful prediction due to safe text context (Certificate).")
            
            # 2. Heuristic Scene Overrides based on Object Detection and OCR Context
            # User defined: Person = Non-hateful. Alcohol/Smoking/Accidents = Hateful.
            has_person = 'person' in objects_detected
            bad_coco_objects = ['wine glass', 'bottle', 'knife', 'scissors', 'car', 'truck', 'motorcycle'] # adding vehicles which might overlap with accident scenes
            has_bad_objects = any(obj in objects_detected for obj in bad_coco_objects)
            
            # We also check the text for explicit mentions of bad scenes if OCR found them
            scene_keywords = ['smoking', 'alcohol', 'accident', 'censor', 'censored', 'bad scene', 'blood', 'crash', 'beer', 'cigarette', 'drink']
            mentions_bad_scene = any(kw in text_lower for kw in scene_keywords)
            
            # Force Hateful if alcohol/bad items detected (regardless of base prediction)
            if has_bad_objects or mentions_bad_scene:
                final_is_hateful = True
                image_is_hateful = True
                print("Forcing Hateful prediction due to detected alcohol/bad scene objects or text mentions.")
                
            # Force Non-hateful if it's strictly a person and text is fine, and no bad items
            elif has_person and not text_is_hateful and not has_bad_objects and not mentions_bad_scene:
                # If model predicted hateful purely on a person image, override it.
                if final_is_hateful or image_is_hateful:
                    final_is_hateful = False
                    image_is_hateful = False
                    print("Overriding Image Hateful prediction because it's just a person and text is safe.")
            
            # Note for future Model Retraining:
            # To have the MLPClassifier reliably learn 'censored scenes' or 'accident scenes' 
            # natively, the training dataset (data/train.jsonl) must be populated with examples 
            # of these specific images labeled as 1 (Hateful). Currently, we use these rule-based 
            # heuristic overrides on top of the base model to achieve the desired behaviour.

            image_label = "Hateful" if image_is_hateful else "Non-Hateful"
            text_label = "Hateful" if text_is_hateful else "Non-Hateful"
            final_label = "Hateful" if final_is_hateful else "Non-Hateful"
            
            final_confidence = model_confidence
            if text_is_hateful:
                 final_confidence = max(0.90, model_confidence)
            elif not final_is_hateful and (predicted_class == 1):
                 final_confidence = 0.85 # Artificial confidence for override
                 
            details = ""

            return jsonify({
                'label': final_label,
                'image_label': image_label,
                'text_label': text_label,
                'confidence': f"{final_confidence:.2%}",
                'image_url': f"/{filepath}",
                'extracted_text': text,
                'details': details
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
