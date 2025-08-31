import requests
import json
from pathlib import Path

def test_api(image_path: str, api_url: str = "http://localhost:8000"):
    """
    Test the crop disease prediction API with an image
    
    Args:
        image_path: Path to the image file to test
        api_url: Base URL of the API (default: http://localhost:8000)
    """
    
    # Check if image file exists
    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}")
        return
    
    # Test health check first
    try:
        health_response = requests.get(f"{api_url}/health")
        if health_response.status_code == 200:
            health_data = health_response.json()
            print("✅ API Health Check:")
            print(f"   Status: {health_data['status']}")
            print(f"   Model Loaded: {health_data['model_loaded']}")
            print(f"   Total Classes: {health_data['total_classes']}")
        else:
            print("❌ API health check failed")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running.")
        return
    
    # Test prediction
    try:
        with open(image_path, 'rb') as image_file:
            files = {'file': image_file}
            response = requests.post(f"{api_url}/predict", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n🔍 Prediction Results for: {image_path}")
            print(f"   Predicted Disease: {result['predicted_class']}")
            print(f"   Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
            print("\n   Top 3 Predictions:")
            for i, pred in enumerate(result['top_predictions'], 1):
                print(f"   {i}. {pred['class']}: {pred['confidence']:.4f} ({pred['confidence']*100:.2f}%)")
        else:
            error_data = response.json()
            print(f"❌ Prediction failed: {error_data['detail']}")
            
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")

def get_model_info(api_url: str = "http://localhost:8000"):
    """Get model information from the API"""
    try:
        response = requests.get(f"{api_url}/model/info")
        if response.status_code == 200:
            info = response.json()
            print("📋 Model Information:")
            print(f"   Type: {info['model_type']}")
            print(f"   Input Shape: {info['input_shape']}")
            print(f"   Output Shape: {info['output_shape']}")
            print(f"   Total Parameters: {info['total_parameters']:,}")
            print(f"   Total Classes: {info['total_classes']}")
        else:
            print("❌ Failed to get model info")
    except Exception as e:
        print(f"❌ Error getting model info: {str(e)}")

def list_classes(api_url: str = "http://localhost:8000"):
    """List all disease classes"""
    try:
        response = requests.get(f"{api_url}/classes")
        if response.status_code == 200:
            data = response.json()
            print(f"\n📝 Available Disease Classes ({data['total_classes']} total):")
            for i, class_name in enumerate(data['classes'], 1):
                crop, disease = class_name.split('___')
                print(f"   {i:2d}. {crop} - {disease}")
        else:
            print("❌ Failed to get class list")
    except Exception as e:
        print(f"❌ Error getting class list: {str(e)}")

if __name__ == "__main__":
    # Example usage
    print("🌱 Crop Disease Prediction API Client")
    print("=" * 40)
    
    # Get model info
    get_model_info()
    
    # List available classes
    list_classes()
    
    # Test with an image (update this path to your test image)
    test_image_path = "test_image.jpg"  # Replace with actual image path
    
    if Path(test_image_path).exists():
        test_api(test_image_path)
    else:
        print(f"\n⚠️  Test image not found: {test_image_path}")
        print("   Update the 'test_image_path' variable with a valid image path to test predictions.")
    
    print(f"\n💡 You can also test the API by visiting: http://localhost:8000")
    print(f"   Or use the interactive docs at: http://localhost:8000/docs")
