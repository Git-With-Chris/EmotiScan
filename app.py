import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from streamlit_option_menu import option_menu
import tensorflow_addons as tfa

# Load the trained model
model_path = "./models/emotiscan_model.h5"
model = load_model(model_path)

# Define class labels
class_labels_emotion = ['negative', 'happy', 'surprised']
class_labels_facs = ['AU17', 'AU1', 'AU2', 'AU25', 'AU27', 'AU4', 'AU7', 'AU23', 'AU24', 'AU6', 'AU12', 'AU15', 'AU14', 'AU11', 'AU26']

# Function to preprocess the image
def preprocess_image(img_array, target_size=(224, 224)):
    # Convert from BGR to RGB
    image = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    # Resize image to target size
    image = cv2.resize(image, target_size)

    # Normalize image
    image = image.astype(np.float32) / 255.0
    
    # Expand dimensions to create a batch of size 1
    image = np.expand_dims(image, axis=0)

    return image

# Function to predict emotion and FACS codes
def predict_emotion_and_facs(img_array):
    img_array = preprocess_image(img_array)
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

# Sidebar - Buttons for navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        menu_icon='list',
        options=['Home', 'About'],
        icons=["house", "pen"],
        default_index=0
    )

# Streamlit app
st.title("EmotiScan 🫥")

# Home page (default)
if selected == "Home":
    st.write("Upload an image and let EmotiScan take a fun guess at the emotion/expression in the image!")

    uploaded_file = st.file_uploader("📸 Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert the file to an opencv image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_array = cv2.imdecode(file_bytes, 1)
        
        # Predict emotion and FACS codes
        predicted_emotion, predicted_facs = predict_emotion_and_facs(img_array)
        
        # Use columns to display predictions to the right of the image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Display the uploaded image with a fixed width
            st.image(img_array, channels="BGR", caption="Your uploaded image 😎", width=320)
        
        with col2:
            # Display predictions in a styled container
            st.write("**Predicted Emotion:**")
            st.info(predicted_emotion.capitalize())
            st.write("**Predicted FACS Codes:**")
            st.info(", ".join(predicted_facs))
        
        st.success('Prediction completed successfully! 🎉')  # Just for fun

# About page
elif selected == "About":
    st.link_button("Github", "https://github.com/Git-With-Chris/EmotiScan")
    st.markdown("### About")
    st.markdown("""
    Welcome to EmotiScan, where we blend the magic of machine learning to understanding emotions and expressions 😁

    #### What is EmotiScan?

    EmotiScan is a fun web app designed to predict emotions and Facial Action Coding System (FACS) codes from images. Using a powerful deep learning model, EmotiScan can analyze your photos and provide insightful predictions about the emotions and facial actions they contain.

    #### How Does It Work?

    1. **Upload Your Image**: Simply upload a photo of a face.
    2. **Magic Happens**: Our pre-trained model gets to work, analyzing the facial features.
    3. **Get Results**: In no time, you'll see a prediction of the emotion and FACS codes associated with the facial expressions in the image.
    """)

    # Add an image to the about section
    st.image("./images/EDAPicture.png", caption="EmotiScan in Action", use_column_width=True)             
    
    st.markdown("""
    #### Why Use EmotiScan?

    - **Fun and Insightful**: Ever wondered what emotions are truly reflected in your selfies? EmotiScan gives you a fun and detailed breakdown.
    - **Advanced Technology**: Built using state-of-the-art deep learning techniques, our model ensures accurate and reliable predictions.
    - **Easy to Use**: With a simple and intuitive interface, anyone can use EmotiScan to explore the world of facial expressions.

    #### Meet the Developer

    EmotiScan was developed by Chris, a passionate developer with expertise in machine learning. Check out his portfolio [here](https://findingchris.netlify.app/)

    So, go ahead, give it a try, and discover the hidden emotions in your photos with EmotiScan! 📸✨

    [GitHub Repository](https://github.com/Git-With-Chris/EmotiScan)

    """)


