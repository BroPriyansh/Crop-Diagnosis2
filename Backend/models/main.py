from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import uvicorn
from typing import Dict, List
import os
import re
from contextlib import asynccontextmanager

# Load model at startup
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    print("Model loaded successfully")
    yield
    # Shutdown
    print("Shutting down...")

# Initialize FastAPI app
app = FastAPI(
    title="Plant Disease Recognition API",
    description="An API for predicting plant diseases from images using deep learning",
    version="1.0.0",
    lifespan=lifespan
)

# Tensorflow Model Prediction
def model_prediction(image_data):
    """Model prediction function adapted from Streamlit version"""
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_data))
    
    # Resize to target size (128, 128) as per your model
    image = image.resize((128, 128))
    
    # Convert to array and preprocess
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    
    # Make prediction
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

# Class names from your Streamlit code
class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
            'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
            'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
              'Tomato___healthy']

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve a simple HTML interface for testing"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Crop Disease Prediction</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .upload-area { border: 2px dashed #ccc; padding: 20px; text-align: center; margin: 20px 0; }
            .result { margin-top: 20px; padding: 15px; background-color: #f5f5f5; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>🌱 Crop Disease Prediction API</h1>
        <p>Upload an image of a crop leaf to predict potential diseases.</p>
        
        <div class="upload-area">
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" id="imageFile" accept="image/*" required>
                <br><br>
                <button type="submit">Predict Disease</button>
            </form>
        </div>
        
        <div id="result" class="result" style="display: none;">
            <h3>Prediction Result:</h3>
            <div id="prediction"></div>
        </div>

        <script>
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const fileInput = document.getElementById('imageFile');
                const file = fileInput.files[0];
                
                if (!file) {
                    alert('Please select an image file');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        document.getElementById('prediction').innerHTML = `
                            <strong>Predicted Disease:</strong> ${result.predicted_class}<br>
                            <strong>Confidence:</strong> ${(result.confidence * 100).toFixed(2)}%<br>
                            <strong>Top 3 Predictions:</strong>
                            <ul>
                                ${result.top_predictions.map(pred => 
                                    `<li>${pred.class}: ${(pred.confidence * 100).toFixed(2)}%</li>`
                                ).join('')}
                            </ul>
                        `;
                        document.getElementById('result').style.display = 'block';
                    } else {
                        alert('Error: ' + result.detail);
                    }
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            });
        </script>
    </body>
    </html>
    """
    return html_content

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    """
    Predict crop disease from uploaded image
    
    Args:
        file: Uploaded image file
    
    Returns:
        Dictionary containing prediction results with validation
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image data
        image_data = await file.read()
        
        # Get prediction using the exact same function as Streamlit
        result_index = model_prediction(image_data)
        predicted_class = class_name[result_index]
        
        # Get full prediction probabilities for confidence and top predictions
        image = Image.open(io.BytesIO(image_data))
        image = image.resize((128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.expand_dims(input_arr, axis=0)
        predictions = model.predict(input_arr)
        
        confidence = float(predictions[0][result_index])
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_predictions = []
        
        for idx in top_3_indices:
            class_confidence = float(predictions[0][idx])
            top_predictions.append({
                "class": class_name[idx],
                "confidence": class_confidence
            })
        
        # Extract true label from filename (same logic as Streamlit)
        prefix_to_class = {
            "AppleCedarRust":    "Apple___Cedar_apple_rust",
            "AppleScab":         "Apple___Apple_scab",
            "CornCommonRust":    "Corn_(maize)___Common_rust_",
            "PotatoEarlyBlight": "Potato___Early_blight",
            "PotatoHealthy":     "Potato___healthy",
            "TomatoEarlyBlight": "Tomato___Early_blight",
            "TomatoHealthy":     "Tomato___healthy",
            "TomatoYellowCurlVirus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
        }
        
        # Get filename and extract prefix
        filename = file.filename
        stem = os.path.splitext(filename)[0]  # Remove extension
        prefix = re.sub(r"\d+$", "", stem)    # Remove trailing digits
        
        # Get true class from filename
        true_class = prefix_to_class.get(prefix, None)
        
        # Validation result
        validation_result = None
        if true_class:
            if predicted_class == true_class:
                validation_result = {
                    "status": "correct",
                    "message": "✅ CORRECT! Model prediction matches expected class",
                    "true_class": true_class
                }
            else:
                validation_result = {
                    "status": "incorrect",
                    "message": "❌ WRONG! Prediction doesn't match expected class",
                    "true_class": true_class
                }
        else:
            validation_result = {
                "status": "unknown",
                "message": "Could not extract true label from filename - unable to validate prediction",
                "true_class": None
            }
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top_predictions": top_predictions,
            "validation": validation_result,
            "filename": filename,
            "model_info": {
                "total_classes": len(class_name),
                "input_shape": model.input_shape
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_type": "Keras Sequential/Functional Model",
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "total_parameters": model.count_params(),
        "total_classes": len(class_name),
        "class_names": class_name
    }

@app.get("/classes")
async def get_classes():
    """Get list of all disease classes the model can predict"""
    return {
        "total_classes": len(class_name),
        "classes": class_name
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "total_classes": len(class_name)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
