from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

model = load_model("/home/manishji/Smart_Mathematics_Tutor/Flask/shape.h5")
print("Model loaded successfully!")

def predict(InputImg):
    img = image.load_img(InputImg, target_size=(64, 64))  # Load and resize the image
    x = image.img_to_array(img)  # Convert image to array
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    x = x / 255.0  # Normalize the image data (if the model expects this)
    pred = model.predict(x)  # Get prediction probabilities
    
    # Print out the probabilities for debugging
    print("Prediction probabilities:", pred)
    
    pred_class = np.argmax(pred, axis=-1)  # Get the index of the highest probability
    print("Predicted class index:", pred_class[0])
    
    # Map the index to the actual class label
    index = ['circle', 'square', 'triangle']  # Class labels
    result = str(index[pred_class[0]])  # Get the predicted label
    return result
result =predict(InputImg="/home/manishji/Downloads/131.png")
print(result)