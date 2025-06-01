import os
import cv2
import torch, numpy as np
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from typing import List
from torchvision.models.video import swin3d_t

class ViolenceDetector:
    def __init__(
        self,
        model_path: str = "VD_classification.pt",
        frame_count: int = 16,
        threshold: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.frame_count = frame_count
        self.threshold = threshold
        self.device = device
        self.transforms = self._get_transforms()
        self.model = self._load_model(model_path)

    def _get_transforms(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def _load_model(self, model_path: str) -> nn.Module:
        model = VDC(num_classes=1).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True), strict=False)
        model.eval()
        return model
    def predict_single_frame(self, frame: np.ndarray) -> dict:
        # Preprocess
        frame_tensor = self.transforms(frame).unsqueeze(0).to(self.device)
        
        # Duplicate frame to simulate temporal dimension (16 frames expected by SWIN3D)
        # Note: This is a workaround - for better accuracy, use actual temporal context
        video_tensor = frame_tensor.unsqueeze(0).repeat(1, 16, 1, 1, 1)
        video_tensor = video_tensor.permute(0, 2, 1, 3, 4)  # [batch, channels, time, height, width]
        
        # Inference
        with torch.no_grad():
            output = self.model(video_tensor)
            prob = torch.sigmoid(output).item()
        
        return {
            "prediction": "Violent" if prob > self.threshold else "Non-Violent",
            "confidence": prob,
            "error": None
        }
    def predict_frame_sequence(self, frames: List[np.ndarray]) -> dict:
        
        # Preprocess all frames
        frame_tensors = [self.transforms(frame) for frame in frames]
        video_tensor = torch.stack(frame_tensors).to(self.device)
        
        # Add batch and temporal dimensions [batch, channels, time, height, width]
        video_tensor = video_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)
        
        # Inference
        with torch.no_grad():
            output = self.model(video_tensor)
            prob = torch.sigmoid(output)       # now in [0, 1]
            prob = prob.item()
        
        return {
            "prediction": "Violent" if prob > self.threshold else "Non-Violent",
            "confidence": prob,
            "error": None
        }

    def extract_frames(self, video_path: str) -> List[torch.Tensor]:
        vid = cv2.VideoCapture(video_path)
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = torch.linspace(0, total_frames - 1, steps=self.frame_count).long().tolist()
        frames = []

        for i in range(total_frames):
            ret, frame = vid.read()
            if not ret:
                break
            if i in frame_indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(self.transforms(frame))

        vid.release()
        return frames

    def predict(self, video_path: str) -> dict:
        try:
            # Extract and preprocess frames
            frames = self.extract_frames(video_path)
            if len(frames) < self.frame_count:
                return {"prediction": "Error", "confidence": 0.0, "error": "Insufficient frames extracted"}

            # Convert to tensor and add batch dimension
            video_tensor = torch.stack(frames).to(self.device)
            video_tensor = video_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [batch, channels, time, height, width]

            # Inference
            with torch.no_grad():
                output = self.model(video_tensor)
                prob = torch.sigmoid(output).item()

            # Classify
            prediction = "Violent" if prob > self.threshold else "Non-Violent"
            return {"prediction": prediction, "confidence": prob, "error": None}

        except Exception as e:
            return {"prediction": "Error", "confidence": 0.0, "error": str(e)}

# Model architecture (must match training)
class VDC(nn.Module):
    def __init__(self, num_classes=1):
        super(VDC, self).__init__()
        self.backbone = swin3d_t(progress=True)
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Custom head
        self.backbone.head = nn.Sequential(
            nn.Linear(768, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Linear(192, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.backbone(x).squeeze(1)

# Example usage in a GUI
if __name__ == "__main__":
    detector = ViolenceDetector(model_path="VD_classification.pt")
    result = detector.predict("V_101.mp4")
    print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.2f})")