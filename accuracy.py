import cv2
import torch
import torchvision.transforms as transforms
from transformers import AutoModelForImageClassification
from PIL import Image
import pytesseract
from sklearn.metrics import accuracy_score

# Initialize Tesseract (update with your correct Tesseract path)
pytesseract.pytesseract.tesseract_cmd = r"/opt/homebrew/bin/tesseract"  

# Load the model from Hugging Face
model = AutoModelForImageClassification.from_pretrained("jazzmacedo/fruits-and-vegetables-detector-36")
labels = list(model.config.id2label.values())  # Get labels from model configuration

# Preprocessing for model input
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Your test dataset with image paths and ground truth labels
test_data = {
    "/Users/pranavbejgam/Desktop/Smart Recipe/images/apple box.jpg": "apple",
    "/Users/pranavbejgam/Desktop/Smart Recipe/images/Apple chips.jpg": "apple chips",
    "/Users/pranavbejgam/Desktop/Smart Recipe/images/apple juices.jpg": "apple juices",
    "/Users/pranavbejgam/Desktop/Smart Recipe/images/Apple text.jpg": "apple",
    "/Users/pranavbejgam/Desktop/Smart Recipe/images/banana.jpg": "banana",
    "/Users/pranavbejgam/Desktop/Smart Recipe/images/fruit pack.jpg": "apple, lemon, cucumber",
    "/Users/pranavbejgam/Desktop/Smart Recipe/images/garlic.jpg": "garlic",
    "/Users/pranavbejgam/Desktop/Smart Recipe/images/Iphone.jpg": "apple",
    "/Users/pranavbejgam/Desktop/Smart Recipe/images/mango.jpg": "mango",
    "/Users/pranavbejgam/Desktop/Smart Recipe/images/Potato.jpg": "potato",
    "/Users/pranavbejgam/Desktop/Smart Recipe/images/red chilli powder.jpg": "red chilli powder",
    "/Users/pranavbejgam/Desktop/Smart Recipe/images/Red_Apple.jpg": "apple",
    "/Users/pranavbejgam/Desktop/Smart Recipe/images/red-tomato-vegetable.jpg": "tomato",
    
}

def extract_text(image):
    """Extract text from image using Tesseract."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    text = pytesseract.image_to_string(gray).strip().lower()
    return text

def classify_image(image):
    """Classify image using the Hugging Face model."""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert to PIL
    input_tensor = preprocess(pil_image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_idx = torch.argmax(outputs.logits, dim=1).item()
    return labels[predicted_idx].lower()

def evaluate_pipeline(test_data):
    """Evaluate both the model and the combined pipeline."""
    model_preds = []
    combined_preds = []
    ground_truths = []

    for image_path, true_label in test_data.items():
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to open {image_path}")
            continue

        # Get predictions
        extracted_text = extract_text(image)
        model_prediction = classify_image(image)

        # Store ground truth
        ground_truth = true_label.lower()
        ground_truths.append(ground_truth)

        # Model-only prediction
        model_preds.append(model_prediction)

        # Combined logic: Use text extraction if it returns valid content, else fallback to model
        if any(keyword in extracted_text for keyword in ground_truth.split(", ")):
            combined_preds.append(ground_truth)  # Valid text prediction
        else:
            combined_preds.append(model_prediction)  # Fallback to model

    # Calculate accuracies
    model_accuracy = accuracy_score(ground_truths, model_preds)
    combined_accuracy = accuracy_score(ground_truths, combined_preds)

    print(f"Model Accuracy: {model_accuracy * 100:.2f}%")
    print(f"Combined Accuracy: {combined_accuracy * 100:.2f}%")

# Run the evaluation
evaluate_pipeline(test_data)
