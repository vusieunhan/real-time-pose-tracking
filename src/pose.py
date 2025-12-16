# src/pose.py

from ultralytics import YOLO
import torch

class PoseEstimator:
    def __init__(self, model_path="yolov8n-pose.pt", device="cuda"):
        self.device = device
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.model.fuse()



    def estimate(self, image, conf=0.5):
        with torch.no_grad():
            results = self.model(image, conf=conf)

        if results[0].keypoints is None:
            return None

        return results[0].keypoints.xy.cpu().numpy()

