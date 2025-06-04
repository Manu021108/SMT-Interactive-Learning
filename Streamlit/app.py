import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Shape formulas dictionary
SHAPE_FORMULAS = {
    'circle': {
        'name': 'Circle',
        'area': 'Area = π × r²',
        'perimeter': 'Circumference = 2 × π × r',
        'description': 'Where r is the radius of the circle',
        'additional_info': [
            'Diameter = 2 × r',
            'π ≈ 3.14159',
            'Sector Area = (θ/360°) × π × r²',
            'Arc Length = (θ/360°) × 2 × π × r'
        ]
    },
    'square': {
        'name': 'Square',
        'area': 'Area = side²',
        'perimeter': 'Perimeter = 4 × side',
        'description': 'Where side is the length of one side of the square',
        'additional_info': [
            'Diagonal = side × √2',
            'All sides are equal',
            'All angles are 90°',
            'Area can also be: Area = (diagonal²)/2'
        ]
    },
    'triangle': {
        'name': 'Triangle',
        'area': 'Area = (1/2) × base × height',
        'perimeter': 'Perimeter = side₁ + side₂ + side₃',
        'description': 'Where base and height are perpendicular measurements',
        'additional_info': [
            "Heron's Formula: Area = √[s(s-a)(s-b)(s-c)]",
            "Where s = (a+b+c)/2 (semi-perimeter)",
            "For right triangle: Area = (1/2) × leg₁ × leg₂",
            "Sum of all angles = 180°"
        ]
    }
}

@st.cache_resource
def load_shape_model():
    """Load the trained model with error handling"""
    model_path = "/home/manishji/SMT-Interactive-Learning/Streamlit/shapes.h5"
    
    try:
        # Check if file exists
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            st.info("Please check the file path and ensure the model file exists.")
            return None
            
        # Try to load the model
        model = load_model(model_path)
        st.success("✅ Model loaded successfully!")
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Possible solutions:")
        st.write("1. Check if the file path is correct")
        st.write("2. Ensure the .h5 file is not corrupted")
        st.write("3. Verify the file has proper read permissions")
        st.write("4. Try re-training and saving the model")
        return None

def predict_shape(input_img, model):
    """Predict the shape from the input image"""
    try:
        # Load and preprocess the image
        img = image.load_img(input_img, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0  # Normalize pixel values
        
        # Make prediction
        pred = model.predict(x, verbose=0)
        pred_class = np.argmax(pred, axis=-1)
        
        # Class labels (make sure this matches your model's training)
        class_labels = ['circle', 'square', 'triangle']
        
        # Get prediction confidence
        confidence = float(np.max(pred))
        predicted_shape = class_labels[pred_class[0]]
        
        return predicted_shape, confidence
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, 0

def display_shape_formulas(shape_name):
    """Display mathematical formulas for the predicted shape"""
    if shape_name in SHAPE_FORMULAS:
        formula_data = SHAPE_FORMULAS[shape_name]
        
        st.markdown(f"## 📐 {formula_data['name']} Formulas")
        
        # Main formulas
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📏 Area Formula")
            st.markdown(f"**{formula_data['area']}**")
            
        with col2:
            st.markdown("### 📐 Perimeter Formula")
            st.markdown(f"**{formula_data['perimeter']}**")
        
        st.markdown(f"*{formula_data['description']}*")
        
        # Additional information
        st.markdown("### 📚 Additional Information")
        for info in formula_data['additional_info']:
            st.markdown(f"• {info}")
            
        # Example calculation section
        st.markdown("### 🧮 Example Calculation")
        show_example_calculation(shape_name)

def show_example_calculation(shape_name):
    """Show example calculations for each shape"""
    if shape_name == 'circle':
        st.markdown("""
        **Example: Circle with radius = 5 units**
        - Area = π × 5² = π × 25 ≈ 78.54 square units
        - Circumference = 2 × π × 5 ≈ 31.42 units
        """)
    elif shape_name == 'square':
        st.markdown("""
        **Example: Square with side = 6 units**
        - Area = 6² = 36 square units
        - Perimeter = 4 × 6 = 24 units
        - Diagonal = 6 × √2 ≈ 8.49 units
        """)
    elif shape_name == 'triangle':
        st.markdown("""
        **Example: Triangle with base = 8 units, height = 6 units**
        - Area = (1/2) × 8 × 6 = 24 square units
        
        **Right Triangle with legs = 3, 4 units**
        - Area = (1/2) × 3 × 4 = 6 square units
        - Hypotenuse = √(3² + 4²) = 5 units
        """)

# Streamlit App
def main():
    st.set_page_config(page_title="Shape Recognition & Math Tutor", page_icon="📐", layout="wide")
    
    st.title("📐 Math Tutor for Shapes")
    st.markdown("### Draw or upload a shape (Circle, Square, Triangle) to get its mathematical formulas!")
    
    # Load model
    model = load_shape_model()
    
    if model is None:
        st.stop()
    
    # Sidebar with instructions
    with st.sidebar:
        st.markdown("## 📋 Instructions")
        st.markdown("""
        1. **Upload an image** of a shape (circle, square, or triangle)
        2. The AI will **predict** the shape
        3. View the **mathematical formulas** for that shape
        4. Learn with **example calculations**
        """)
        
        st.markdown("## 🎨 Supported Shapes")
        st.markdown("• Circle ⭕")
        st.markdown("• Square ⬜")
        st.markdown("• Triangle 🔺")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of a circle, square, or triangle"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🖼️ Uploaded Image")
            image_data = Image.open(uploaded_file)
            st.image(image_data, caption="Your Drawing", use_column_width=True)
            
            # Save image temporarily for prediction
            temp_image_path = "temp_shape.png"
            image_data.save(temp_image_path)
        
        with col2:
            st.markdown("### 🤖 AI Prediction")
            
            if st.button("🔍 Predict Shape", type="primary"):
                with st.spinner("Analyzing your shape..."):
                    predicted_shape, confidence = predict_shape(temp_image_path, model)
                    
                    if predicted_shape:
                        st.success(f"**Predicted Shape: {predicted_shape.title()}**")
                        st.info(f"Confidence: {confidence:.2%}")
                        
                        # Display formulas
                        display_shape_formulas(predicted_shape)
                    
                # Clean up temporary file
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
    
    else:
        st.info("👆 Please upload an image of a shape to get started!")
        
        # Show sample formulas
        st.markdown("## 📚 Shape Formulas Reference")
        
        tabs = st.tabs(["Circle ⭕", "Square ⬜", "Triangle 🔺"])
        
        with tabs[0]:
            display_shape_formulas('circle')
        
        with tabs[1]:
            display_shape_formulas('square')
            
        with tabs[2]:
            display_shape_formulas('triangle')

if __name__ == "__main__":
    main()