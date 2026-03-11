

"""Inference endpoint for federated learning model."""

import torch
from pytorchexample.task import Net
from storage.DB import MinIOClient
from torchvision import transforms
from PIL import Image
import io


class ModelInference:
    """Class to handle model inference from MinIO."""
    
    def __init__(self, model_path: str = None):
        """
        Initialize inference handler.
        
        Args:
            model_path: Path to model in MinIO (e.g., "models/final_model_20260311_143020.pt")
                       If None, will use the latest model
        """
        self.minio_client = MinIOClient()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_path = model_path
        
        # Define image transforms (same as training)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
    def load_model(self):
        """Load model from MinIO."""
        if self.model_path is None:
            # Get latest model
            self.model_path = self._get_latest_model()
        
        print(f"Loading model from MinIO: {self.model_path}")
        
        # Download model from MinIO
        state_dict = self.minio_client.load_model(self.model_path)
        
        if state_dict is None:
            raise ValueError(f"Failed to load model from {self.model_path}")
        
        # Initialize model and load weights
        self.model = Net()
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded successfully on {self.device}")
        
    def _get_latest_model(self):
        """Get the latest model from MinIO bucket."""
        try:
            objects = self.minio_client.client.list_objects(
                self.minio_client.bucket_name,
                prefix="models/",
                recursive=True
            )
            
            models = [obj.object_name for obj in objects if obj.object_name.endswith('.pt')]
            
            if not models:
                raise ValueError("No models found in MinIO bucket")
            
            # Sort by name (timestamp-based) and get latest
            latest_model = sorted(models)[-1]
            return latest_model
            
        except Exception as e:
            raise ValueError(f"Error finding latest model: {e}")
    
    def predict_image(self, image_path: str):
        """
        Predict single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            predicted_class: int
            confidence: float
        """
        if self.model is None:
            self.load_model()
        
        # Load and preprocess image
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        return predicted.item(), confidence.item()
    
    def predict_tensor(self, tensor: torch.Tensor):
        """
        Predict from tensor.
        
        Args:
            tensor: Image tensor (1, 28, 28) or (batch, 1, 28, 28)
            
        Returns:
            predictions: list of (class, confidence) tuples
        """
        if self.model is None:
            self.load_model()
        
        # Ensure batch dimension
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        tensor = tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, 1)
        
        results = list(zip(predicted.cpu().numpy(), confidences.cpu().numpy()))
        return results


def main():
    """Example usage."""
    inferencer = ModelInference()
    
    inferencer.load_model()
    
    try:
        image_path = "test.png"  # Replace with your image
        predicted_class, confidence = inferencer.predict_image(image_path)
        print(f"\nPrediction: Class {predicted_class} (Confidence: {confidence:.4f})")
    except FileNotFoundError:
        print("\nNo test image found. Skipping image prediction.")
    


if __name__ == "__main__":
    main()