# MultipleFiles/main.py
# (No changes needed, keep as is from previous response)
import os
from contextlib import asynccontextmanager
from typing import List, Dict, Any

import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io # Import io module for BytesIO
from fastapi.responses import HTMLResponse # Import HTMLResponse

# Load the TensorFlow Keras model
model = None
class_name: List[str] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, class_name
    # Load the model
    model_path = "trained_plant_disease_model.keras"
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found at {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")

    # Define class names (ensure this matches your model's output)
    class_name = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Blueberry___healthy', 'Cherry_(sour)___Powdery_mildew', 'Cherry_(sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust',
        'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
        'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
        'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
        'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
        'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___healthy'
    ]
    print(f"Loaded {len(class_name)} classes.")
    yield
    # Clean up the model resources (optional)
    print("Shutting down model server.")

app = FastAPI(
    title="Plant Disease Prediction API",
    description="API for predicting plant diseases from images using a TensorFlow Keras model.",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Pydantic model for prediction response
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    top_predictions: List[Dict[str, Any]]
    validation: Dict[str, Any]
    filename: str
    model_info: Dict[str, Any]

# Helper function to preprocess image and make prediction
async def model_prediction(image_bytes: bytes) -> Dict[str, Any]:
    if model is None or not class_name:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please wait for startup or check server logs."
        )
    
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Resize image to model's expected input size (e.g., 128x128)
        # Assuming model expects (128, 128, 3)
        input_shape = model.input_shape[1:3] # Get H, W from (None, H, W, C)
        image = image.resize(input_shape)
        
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Ensure image has 3 channels (RGB)
        if img_array.shape[-1] == 4: # If RGBA, convert to RGB
            img_array = img_array[..., :3]
        elif img_array.shape[-1] == 1: # If grayscale, convert to RGB
            img_array = np.stack([img_array[:,:,0]]*3, axis=-1)
        
        # Normalize pixel values if your model expects it (e.g., 0-1 or -1 to 1)
        # Assuming model was trained with pixel values scaled to 0-1
        img_array = img_array / 255.0
        
        # Expand dimensions to create a batch of 1 image (1, 128, 128, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = model.predict(img_array)
        
        # Get the predicted class index and confidence
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        predicted_class_name = class_name[predicted_class_index]

        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1] # Get indices of top 3, in descending order
        top_predictions = []
        for i in top_3_indices:
            top_predictions.append({
                "class": class_name[i],
                "confidence": float(predictions[0][i])
            })

        return {
            "predicted_class_index": predicted_class_index,
            "predicted_class_name": predicted_class_name,
            "confidence": confidence,
            "top_predictions": top_predictions,
            "model_input_shape": model.input_shape[1:],
            "total_classes": len(class_name)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during model prediction: {e}"
        )

@app.get("/", summary="Root endpoint for web UI")
async def read_root():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Plant Disease Predictor</title>
        <style>
            body { font-family: sans-serif; margin: 2em; background-color: #f4f4f4; color: #333; }
            h1 { color: #4CAF50; }
            .container { background-color: #fff; padding: 2em; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); max-width: 600px; margin: 2em auto; }
            input[type="file"] { margin-bottom: 1em; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }
            button { background-color: #4CAF50; color: white; padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; font-size: 1em; }
            button:hover { background-color: #45a049; }
            #result { margin-top: 1.5em; padding: 1em; border: 1px solid #eee; border-radius: 4px; background-color: #e8f5e8; }
            #result h2 { color: #2e7d32; margin-top: 0; }
            #result p { margin: 0.5em 0; }
            .error { color: red; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Upload Image for Plant Disease Prediction</h1>
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required>
                <button type="submit">Predict Disease</button>
            </form>
            <div id="result">
                <h2>Prediction Result:</h2>
                <p id="predictedClass"></p>
                <p id="confidence"></p>
                <p id="topPredictions"></p>
                <p id="validation"></p>
                <p id="filename"></p>
                <p id="modelInfo"></p>
            </div>
        </div>

        <script>
            document.getElementById('uploadForm').addEventListener('submit', async function(event) {
                event.preventDefault();
                const formData = new FormData(this);
                
                document.getElementById('predictedClass').textContent = 'Predicting...';
                document.getElementById('confidence').textContent = '';
                document.getElementById('topPredictions').textContent = '';
                document.getElementById('validation').textContent = '';
                document.getElementById('filename').textContent = '';
                document.getElementById('modelInfo').textContent = '';

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.detail || 'Prediction failed');
                    }

                    const data = await response.json();
                    document.getElementById('predictedClass').textContent = 'Predicted Class: ' + data.predicted_class;
                    document.getElementById('confidence').textContent = 'Confidence: ' + (data.confidence * 100).toFixed(2) + '%';
                    
                    let topPredsHtml = 'Top 3 Predictions: ';
                    data.top_predictions.forEach((pred, index) => {
                        topPredsHtml += `${pred.class} (${(pred.confidence * 100).toFixed(2)}%)`;
                        if (index < data.top_predictions.length - 1) {
                            topPredsHtml += ', ';
                        }
                    });
                    document.getElementById('topPredictions').textContent = topPredsHtml;

                    document.getElementById('validation').textContent = 'Validation: ' + JSON.stringify(data.validation);
                    document.getElementById('filename').textContent = 'Filename: ' + data.filename;
                    document.getElementById('modelInfo').textContent = 'Model Info: Total Classes=' + data.model_info.total_classes + ', Input Shape=' + data.model_info.input_shape;

                } catch (error) {
                    document.getElementById('predictedClass').textContent = 'Error: ' + error.message;
                    document.getElementById('predictedClass').classList.add('error');
                    console.error('Error:', error);
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/predict", response_model=PredictionResponse, summary="Predict disease from an uploaded image")
async def predict_disease(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not an image."
        )
    
    image_bytes = await file.read()
    prediction_results = await model_prediction(image_bytes)

    # Attempt to extract true label from filename for validation
    true_label = "unknown"
    filename_lower = file.filename.lower()
    for cls in class_name:
        cls_lower = cls.lower().replace("___", "").replace("_", "")
        if cls_lower in filename_lower:
            true_label = cls
            break
    
    validation_status = "N/A"
    if true_label != "unknown":
        validation_status = "Correct" if prediction_results["predicted_class_name"] == true_label else "Incorrect"

    return PredictionResponse(
        predicted_class=prediction_results["predicted_class_name"],
        confidence=prediction_results["confidence"],
        top_predictions=prediction_results["top_predictions"],
        validation={
            "true_label": true_label,
            "status": validation_status
        },
        filename=file.filename,
        model_info={
            "total_classes": prediction_results["total_classes"],
            "input_shape": prediction_results["model_input_shape"]
        }
    )

@app.get("/model/info", summary="Get information about the loaded model")
async def get_model_info():
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded."
        )
    return {
        "model_type": "TensorFlow Keras",
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "total_parameters": model.count_params(),
        "total_classes": len(class_name),
        "class_names_sample": class_name[:5] + ["..."] + class_name[-5:] if len(class_name) > 10 else class_name
    }

@app.get("/classes", summary="Get all disease class names")
async def get_classes():
    if not class_name:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Class names not loaded."
        )
    return {"classes": class_name}

@app.get("/health", summary="Health check endpoint")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "total_classes": len(class_name) if class_name else 0
    }

if __name__ == "__main__":
    import uvicorn
    # from fastapi.responses import HTMLResponse # Already imported above

    # Check if the model file exists before starting
    if not os.path.exists("trained_plant_disease_model.keras"):
        print("Error: 'trained_plant_disease_model.keras' not found.")
        print("Please ensure the model file is in the same directory as main.py.")
        exit(1) # Exit if model is not found

    uvicorn.run(app, host="0.0.0.0", port=8000)