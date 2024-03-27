# models/wall_detection_model.py
import torch
import torchvision.transforms as transforms
from torchvision import models

# Define your wall detection model
class WallDetectionModel:
    def __init__(self, model_path):
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=False)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def detect_walls(self, image):
        # Preprocess the image
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
            predicted_class = output.argmax(0)

        # Example post-processing logic (modify as needed)
        walls = (predicted_class == 1).cpu().numpy().astype(int)
        return walls
