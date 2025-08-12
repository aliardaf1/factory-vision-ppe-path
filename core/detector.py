# core/detector.py
from typing import List, Tuple
import numpy as np
from abc import ABC, abstractmethod

Box = Tuple[int, int, int, int]  # x1, y1, x2, y2

class Detection:
    def __init__(self, box: Box, cls_id: int, score: float):
        self.box = box
        self.cls_id = cls_id
        self.score = score

class Detector(ABC):
    @abstractmethod
    def load(self, weights_path: str, **kwargs): ...
    @abstractmethod
    def predict(self, frame: np.ndarray, **kwargs) -> List[Detection]: ...
    @abstractmethod
    def class_names(self) -> dict: ...