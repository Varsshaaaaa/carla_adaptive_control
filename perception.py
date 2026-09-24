import torch
from ultralytics import YOLO
import config

class PerceptionSystem:
    def __init__(self, model_name='yolov8n.pt'):
        # Check if CUDA is available, otherwise use CPU
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Loading YOLO model {model_name} on {self.device}...")
        self.model = YOLO(model_name)
        
        # Classes we care about: 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
        self.target_classes = [0, 1, 2, 3, 5, 7]
        self.conf_threshold = config.YOLO_CONF_THRESHOLD

    def detect(self, bgr_image):
        """
        Runs YOLO inference on a BGR image and returns structured detections.
        """
        results = self.model(bgr_image, verbose=False, device=self.device, 
                             classes=self.target_classes, conf=self.conf_threshold)
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                
                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })
        
        return detections
