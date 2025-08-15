# detectors/yolo_v8.py
from ultralytics import YOLO
import torch
import numpy as np

class YOLOv8Detector:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.names = {}

    def load(self, weights: str, device: str | None = None):
        if device:                       # çağrıdan geleni kullan
            self.device = device
        self.model = YOLO(weights)
        # modele device ata
        self.model.to(self.device)
        # fp16 (yalnızca cuda)
        self.half = self.device.startswith("cuda")
        if self.half:
            try:
                self.model.model.half()
            except Exception:
                pass
        # sınıf isimleri
        self.names = getattr(self.model.model, "names", getattr(self.model, "names", {}))
        # debug
        p = next(self.model.model.parameters())
        print(f"[YOLOv8Detector] device={self.device} param_device={p.device} dtype={p.dtype}")

    def class_names(self):
        return self.names

    def predict(self, frame_bgr, conf=0.35, imgsz=640):
        # Ultralytics'e device ve half param'larını ver
        res = self.model.predict(
            frame_bgr,
            imgsz=imgsz,
            conf=conf,
            device=0 if self.device.startswith("cuda") else "cpu",
            verbose=False,
            half=self.half
        )
        # kendi Detection tipine çeviriyorsan eski mantığını aynen uygula
        detections = []
        for r in res:
            for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = map(int, b)
                detections.append(Detection(box=(x1, y1, x2, y2), cls_id=int(c), score=float(s)))
        return detections

# Detection kendi sınıfınsa burada veya ayrı dosyada:
class Detection:
    def __init__(self, box, cls_id, score):
        self.box = box
        self.cls_id = cls_id
        self.score = score
