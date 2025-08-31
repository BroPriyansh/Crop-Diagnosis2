# Crop Disease Prediction API

A FastAPI application that predicts crop diseases from leaf images using a trained deep learning model.

## Features

- 🌱 Predict crop diseases from uploaded images
- 📊 Get confidence scores and top-3 predictions
- 🔍 Model information and health check endpoints
- 🖥️ Simple web interface for testing
- 📚 Automatic API documentation with Swagger UI

## Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the model file `trained_plant_disease_model.keras` is in the same directory as `main.py`

3. Run the FastAPI application:
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Usage

### Web Interface
Open your browser and navigate to `http://localhost:8000` to access the simple web interface for testing.

### API Endpoints

#### 1. Predict Disease
- **Endpoint:** `POST /predict`
- **Description:** Upload an image to predict crop disease
- **Input:** Image file (JPEG, PNG, etc.)
- **Output:** JSON with prediction results

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -F "file=@path/to/your/image.jpg"
```

**Response:**
```json
{
  "predicted_class": "Tomato___Late_blight",
  "confidence": 0.95,
  "top_predictions": [
    {
      "class": "Tomato___Late_blight",
      "confidence": 0.95
    },
    {
      "class": "Tomato___Early_blight",
      "confidence": 0.03
    },
    {
      "class": "Tomato___healthy",
      "confidence": 0.02
    }
  ],
  "model_info": {
    "total_classes": 38,
    "input_shape": [null, 224, 224, 3]
  }
}
```

#### 2. Model Information
- **Endpoint:** `GET /model/info`
- **Description:** Get detailed information about the loaded model

#### 3. Disease Classes
- **Endpoint:** `GET /classes`
- **Description:** Get list of all disease classes the model can predict

#### 4. Health Check
- **Endpoint:** `GET /health`
- **Description:** Check if the API and model are working properly

### API Documentation
Once the server is running, you can access:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

## Supported Crop Diseases

The model can predict diseases for the following crops:
- 🍎 Apple (Apple scab, Black rot, Cedar apple rust)
- 🫐 Blueberry
- 🍒 Cherry (Powdery mildew)
- 🌽 Corn/Maize (Cercospora leaf spot, Common rust, Northern Leaf Blight)
- 🍇 Grape (Black rot, Esca, Leaf blight)
- 🍊 Orange (Citrus greening)
- 🍑 Peach (Bacterial spot)
- 🌶️ Bell Pepper (Bacterial spot)
- 🥔 Potato (Early blight, Late blight)
- 🍓 Strawberry (Leaf scorch)
- 🍅 Tomato (Multiple diseases including blight, mold, viruses)
- And more...

## Model Details

- **Input Size:** 128x128 RGB images
- **Total Classes:** 38 different disease/healthy classes
- **Model Type:** Keras deep learning model
- **Preprocessing:** Images are automatically resized and normalized

## Error Handling

The API includes comprehensive error handling for:
- Invalid file formats
- Model loading issues
- Image processing errors
- Server errors

## Development

### Running in Development Mode
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Testing the API
You can test the API using:
1. The built-in web interface at `http://localhost:8000`
2. Swagger UI at `http://localhost:8000/docs`
3. Command line tools like curl or httpie
4. Python requests library

### Customizing Class Names
If your model was trained on different classes, update the `DEFAULT_CLASS_NAMES` list in `main.py` to match your model's output classes.

## Troubleshooting

### Common Issues
1. **Model not found:** Ensure `trained_plant_disease_model.keras` is in the same directory
2. **TensorFlow errors:** Make sure you have the correct TensorFlow version installed
3. **Image processing errors:** Ensure uploaded files are valid image formats
4. **Port conflicts:** Change the port in the uvicorn command if 8000 is already in use

### Logs
The application prints startup logs and any errors to the console for debugging.
