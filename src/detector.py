from ultralytics import YOLO
import torch

class PersonDetector:
    def __init__(self, model_path="yolov8n.pt", device="cuda"):
        self.device = device
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def detect(self, frame, conf=0.5):
        """
        Input:
            frame: BGR image (OpenCV)
        Output:
            detections: list of ([x, y, w, h], score, "person")
        """
        results = self.model(frame, conf=conf)

        detections = []

        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        for box, score, cls in zip(boxes, scores, classes):
            if int(cls) != 0:  # only PERSON
                continue

            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1

            # lọc bbox quá nhỏ
            if w * h < 800:
                continue

            detections.append(([x1, y1, w, h], float(score), "person"))

        return detections
