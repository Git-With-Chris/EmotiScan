import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the trained model
model_path = "./Models/emotiscan_model.h5"
model = load_model(model_path)

# Define class labels
class_labels_emotion = ['negative', 'happy', 'surprised']
class_labels_facs = ['AU17', 'AU1', 'AU2', 'AU25', 'AU27', 'AU4', 'AU7', 'AU23', 'AU24', 'AU6', 'AU12', 'AU15', 'AU14', 'AU11', 'AU26']

# Function to preprocess the image
def preprocess_image(img_path, target_size=(224, 224)):
    # Load image using OpenCV
    image = cv2.imread(img_path)
    
    if image is None:
        raise FileNotFoundError(f"Image file not found at: {img_path}")

    # Convert from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image to target size
    image = cv2.resize(image, target_size)

    # Normalize image
    image = image.astype(np.float32) / 255.0
    
    # Expand dimensions to create a batch of size 1
    image = np.expand_dims(image, axis=0)

    return image

# Function to predict emotion and FACS codes
def predict_emotion_and_facs(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)

    # Assuming model outputs facs_predictions first and emotion_predictions second
    facs_predictions, emotion_predictions = predictions[0], predictions[1]

    # Define Threshold for array of probabilities
    threshold = 0.4
    facs_predictions = (facs_predictions > threshold).astype(int)
    
    # For emotion, using argmax to get the predicted class
    predicted_labels_emotion_classes = np.argmax(emotion_predictions, axis=1)
    
    # Get human-readable labels
    predicted_emotion_label = class_labels_emotion[predicted_labels_emotion_classes[0]]
    predicted_facs_labels = [class_labels_facs[i] for i in range(len(class_labels_facs)) if facs_predictions[0][i] == 1]

    return predicted_emotion_label, predicted_facs_labels