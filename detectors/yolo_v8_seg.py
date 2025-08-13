# detectors/yolo_v8_seg.py
from typing import Tuple, List
import numpy as np
from ultralytics import YOLO
import cv2

class YOLOv8Seg:
    def __init__(self):
        self.model = None
        self.names = {}

    def load(self, weights: str = "yolov8n-seg.pt"):
        self.model = YOLO(weights)   # PyTorch tabanlı
        self.names = self.model.names

    def predict_mask(self, frame: np.ndarray, conf: float = 0.25, imgsz: int = 640) -> np.ndarray:
        """ Tek sınıf (yellow_path) varsayımıyla birleşik bir ikili maske döndürür. """
        res = self.model(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
        if res.masks is None:
            return np.zeros(frame.shape[:2], dtype=np.uint8)
        # maskeleri birleştir
        m = res.masks.data.cpu().numpy()  # [N, Hm, Wm] -> 0/1
        # görüntü boyutuna ölçekle
        H, W = frame.shape[:2]
        full = np.zeros((H, W), dtype=np.uint8)
        for mi in m:
            mi = (mi * 255).astype(np.uint8)
            mi = cv2.resize(mi, (W, H), interpolation=cv2.INTER_NEAREST)
            full = np.maximum(full, mi)
        return (full > 127).astype(np.uint8)
