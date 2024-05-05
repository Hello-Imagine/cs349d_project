import torch
from torchvision.transforms import functional as F

class YOLOv5Detector:
    def __init__(self, model_name='yolov5s', device=None):
        # Load the model from torch.hub
        self.model = torch.hub.load('ultralytics/yolov5', model_name)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def detect(self, images):
        # Support both single image and batched images
        if not (isinstance(images, torch.Tensor) and images.dim() == 4) and not isinstance(images, list):
            images = [images]
        
        # Perform inference
        with torch.no_grad():
            results = self.model(images)
        
        # Extract names of detected objects
        # detected_names = []
        # for result in results.pandas().xyxy:
        #     names = result[:, -1].int().tolist()  # Get class IDs as integers
        #     # Convert class IDs to names using model's .names attribute
        #     detected_names.append([self.model.names[i] for i in names])

        return results

# Usage
if __name__ == '__main__':
    # Initialize the detector
    detector = YOLOv5Detector('yolov5s')
    
    # Image path
    image_path = 'data/images/zidane.jpg'
    
    # Detect objects in the image
    detected_objects = detector.detect(image_path)
    print("Detected objects:", detected_objects)
