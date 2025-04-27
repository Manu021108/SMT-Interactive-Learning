import streamlit as st
from PIL import Image, ImageGrab
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import base64
import io

# Load the trained model
model = load_model("/home/manishji/Smart_Mathematics_Tutor/shapes.h5")

# Prediction function
def predict(InputImg):
    try:
        img = image.load_img(InputImg, target_size=(64, 64))  # Load and reshape the image
        x = image.img_to_array(img)  # Convert image to array
        x = np.expand_dims(x, axis=0)  # Add batch dimension
        x = x / 255.0  # Normalize the image data (if the model expects this)
        pred = model.predict(x)  # Get prediction probabilities
        pred_class = np.argmax(pred, axis=-1)  # Get the index of the highest probability
        index = ['circle', 'square', 'triangle']  # Class labels
        result = str(index[pred_class[0]])  # Get the predicted label
        return result
    except Exception as e:
        print("Error during prediction:", e)
        return "Error"

# Streamlit UI
st.title("Maths Tutor for Shapes")

st.markdown("### Draw a shape (Circle, Square, Triangle) to get its formula")

# Display instructions
st.text("Draw a shape and click 'Predict' to see its formula.")

# Set up canvas
st.markdown("### Drawing Canvas")
canvas = st.empty()

# Use the file uploader to upload an image (this will replace Tkinter's drawing function)
uploaded_image = st.file_uploader("Upload your drawing", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    # Process the image
    image_data = Image.open(uploaded_image)
    
    # Display the uploaded image
    st.image(image_data, caption="Uploaded Image", use_column_width=True)

    # Save the image locally for prediction
    image_data.save("dist.png")
    
    # Get prediction
    prediction_result = predict("dist.png")
    
    # Show prediction result
    st.markdown(f"### Prediction: {prediction_result}")

# Optional: Clear the uploaded image (you can add a button to reset the app)
if st.button('Clear Image'):
    st.experimental_rerun()
