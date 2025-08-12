# detectors/yolo_v8.py
from typing import List
import numpy as np
from ultralytics import YOLO
from core.detector import Detector, Detection

class YOLOv8Detector(Detector):
    def __init__(self):
        self.model = None
        self._names = {}

    def load(self, weights_path: str = "yolov8n.pt", **kwargs):
        # yolov8n.pt yazarsan ilk çalıştırmada otomatik indirir
        self.model = YOLO(weights_path)
        self._names = self.model.names

    def predict(self, frame: np.ndarray, conf: float = 0.35, imgsz: int = 640, **kwargs) -> List[Detection]:
        res = self.model(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
        out: List[Detection] = []
        for b in res.boxes:
            cls_id = int(b.cls[0].item())
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()
            score = float(b.conf[0].item())
            out.append(Detection((x1, y1, x2, y2), cls_id, score))
        return out

    def class_names(self) -> dict:
        return self._names
