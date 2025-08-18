from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import os

# ---- Birleşik çıkış tipleri ----
class Detection:
    def __init__(self, xyxy, conf: float, cls_id: int, cls_name: str, source: str):
        self.xyxy = [int(x) for x in xyxy]   # [x1,y1,x2,y2]
        self.conf = float(conf)
        self.cls_id = int(cls_id)
        self.cls_name = str(cls_name)        # örn: "ppe:person"
        self.source = str(source)            # örn: "ppe"

class SegMask:
    def __init__(self, mask: np.ndarray, cls_id: int, cls_name: str, source: str):
        # mask: (H, W) uint8/bool (1=alan, 0=arka plan)
        self.mask = (mask > 0).astype(np.uint8)
        self.cls_id = int(cls_id)
        self.cls_name = str(cls_name)        # örn: "path:walk_zone"
        self.source = str(source)

# ---- Adapter tabanı ----
class BaseAdapter:
    def __init__(self, name_prefix: str, interval: int = 1):
        self.prefix = name_prefix
        self.interval = max(1, int(interval))  # kaç karede bir koşturulacak

    def should_run(self, frame_idx: int) -> bool:
        return (frame_idx % self.interval) == 0

    def predict(self, frame_bgr) -> Dict[str, List]:
        raise NotImplementedError

# ---- Ultralytics YOLO Adapters ----
from ultralytics import YOLO
import torch

def safe_load_yolo(weights_path: str | None,
                   fallback_path: str | None = None,
                   allow_download: bool = False) -> YOLO:
    """
    Sadece yereldeki dosyaları yükler. Yoksa:
      - fallback_path varsa onu yükler
      - yoksa indirmeyi DENEMEZ, açıkça RuntimeError fırlatır.
    """
    def _ok(p): return isinstance(p, str) and os.path.isfile(p)

    if _ok(weights_path):
        return YOLO(weights_path)           # yerel dosya
    if _ok(fallback_path):
        print(f"[safe_load] weights bulunamadı, fallback kullanılıyor: {fallback_path}")
        return YOLO(fallback_path)

    if allow_download:
        return YOLO(weights_path or "yolov8n.pt")

    raise RuntimeError(
        f"[safe_load] Yerel model bulunamadı. weights='{weights_path}', fallback='{fallback_path}'. "
        "İndirme kapalı (allow_download=False)."
    )

class YoloDetAdapter(BaseAdapter):
    """YOLOv8 detection (bbox) modeli için adapter (ör: PPE/person/forklift)"""
    def __init__(self, weights: str, device: Optional[str] = None,
                 name_prefix: str = "ppe", conf: float = 0.35, imgsz: int = 640,
                 interval: int = 1, half: bool = True,
                 fallback_path: Optional[str] = None, allow_download: bool = False):
        super().__init__(name_prefix, interval)
        self.model = safe_load_yolo(weights, fallback_path=fallback_path, allow_download=allow_download)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.conf = conf
        self.imgsz = imgsz
        self.half = half and self.device.startswith("cuda")
        try:
            if self.half:
                self.model.model.half()
        except Exception:
            self.half = False
        self.names = getattr(self.model.model, "names", getattr(self.model, "names", {}))

    def predict(self, frame_bgr):
        out = {"detections": [], "masks": []}
        res = self.model.predict(
            frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            device=0 if self.device.startswith("cuda") else "cpu",
            verbose=False,
            half=self.half
        )
        r = res[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out
        xyxy = r.boxes.xyxy
        conf = r.boxes.conf
        cls  = r.boxes.cls
        # CPU'ya indir
        xyxy = xyxy.detach().cpu().numpy()
        conf = conf.detach().cpu().numpy()
        cls  = cls.detach().cpu().numpy().astype(int)
        for i in range(len(cls)):
            name = self.names.get(int(cls[i]), str(int(cls[i])))
            out["detections"].append(
                Detection(xyxy=xyxy[i], conf=conf[i], cls_id=int(cls[i]),
                          cls_name=f"{self.prefix}:{name}", source=self.prefix)
            )
        return out

class YoloSegAdapter(BaseAdapter):
    """YOLOv8 segmentation (mask) modeli için adapter (ör: walk_zone)"""
    def __init__(self, weights: str, device: Optional[str] = None,
                 name_prefix: str = "path", conf: float = 0.35, imgsz: int = 640,
                 interval: int = 2, half: bool = True,
                 fallback_path: Optional[str] = None, allow_download: bool = False):
        super().__init__(name_prefix, interval)
        self.model = safe_load_yolo(weights, fallback_path=fallback_path, allow_download=allow_download)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.conf = conf
        self.imgsz = imgsz
        self.half = half and self.device.startswith("cuda")
        try:
            if self.half:
                self.model.model.half()
        except Exception:
            self.half = False
        self.names = getattr(self.model.model, "names", getattr(self.model, "names", {}))

    def predict(self, frame_bgr):
        out = {"detections": [], "masks": []}
        res = self.model.predict(
            frame_bgr,
            task="segment",
            imgsz=self.imgsz,
            conf=self.conf,
            device=0 if self.device.startswith("cuda") else "cpu",
            verbose=True,  # Debug için verbose açık
            half=self.half
        )
        r = res[0]
        
        # Debug bilgileri
        print(f"[YoloSegAdapter] Model output type: {type(r)}")
        print(f"[YoloSegAdapter] Has masks: {hasattr(r, 'masks')}")
        print(f"[YoloSegAdapter] Has boxes: {hasattr(r, 'boxes')}")
        if hasattr(r, 'boxes'):
            print(f"[YoloSegAdapter] Number of boxes: {len(r.boxes)}")
        
        if r.masks is None or r.boxes is None or len(r.boxes) == 0:
            return out
        cls = r.boxes.cls.detach().cpu().numpy().astype(int)
        data = r.masks.data.detach().cpu().numpy()  # [N,H,W]
        for i, c in enumerate(cls):
            name = self.names.get(int(c), str(int(c)))
            out["masks"].append(
                SegMask(mask=data[i], cls_id=int(c),
                        cls_name=f"{self.prefix}:{name}", source=self.prefix)
            )
        return out

# ---- Model Manager ----
class ModelManager:
    def __init__(self):
        self.adapters: Dict[str, BaseAdapter] = {}
        self.enabled: Dict[str, bool] = {}

    def register(self, key: str, adapter: BaseAdapter, enabled: bool = True):
        self.adapters[key] = adapter
        self.enabled[key] = enabled

    def set_enabled(self, key: str, value: bool):
        if key in self.enabled:
            self.enabled[key] = bool(value)

    def infer(self, frame_bgr, frame_idx: int = 0):
        dets: List[Detection] = []
        masks: List[SegMask] = []
        for key, ad in self.adapters.items():
            if not self.enabled.get(key, True):
                continue
            if not ad.should_run(frame_idx):
                continue
            res = ad.predict(frame_bgr)
            dets.extend(res.get("detections", []))
            masks.extend(res.get("masks", []))
        return {"detections": dets, "masks": masks}
