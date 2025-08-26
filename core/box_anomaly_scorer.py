# box_anomaly_scorer.py
import os, json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.neighbors import NearestNeighbors

def _list_images(folder: Path) -> List[Path]:
    exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])

class _Encoder(nn.Module):
    def __init__(self, img_size: int = 256):
        super().__init__()
        try:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except Exception:
            self.backbone = models.resnet50(weights=None)
        self.backbone.fc = nn.Identity()
        self.backbone.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

    @torch.no_grad()
    def encode_pil(self, pil: Image.Image) -> np.ndarray:
        x = self.tf(pil).unsqueeze(0).to(self.device)
        z = self.backbone(x).cpu().numpy()
        return z

    @torch.no_grad()
    def encode_batch(self, pil_list: List[Image.Image]) -> np.ndarray:
        x = torch.stack([self.tf(im) for im in pil_list]).to(self.device)
        z = self.backbone(x).cpu().numpy()
        return z

class BoxAnomalyScorer:
    def __init__(
        self,
        artifacts_dir: str = r"C:\Users\Stjya2\OneDrive - ALISAN LOJISTIK A.S\Belgeler\GitHub\factory-vision-ppe-path\results",
        data_dir: Optional[str] = r"C:\Users\Stjya2\OneDrive - ALISAN LOJISTIK A.S\Belgeler\GitHub\factory-vision-ppe-path\anomaly_data\train\good",
        img_size: int = 256,
        k: int = 5,
        pctl: float = 99.0,
    ):
        self.artifacts_dir = Path(artifacts_dir)
        self.data_dir = Path(data_dir) if data_dir else None
        self.img_size = img_size
        self.k = int(k)
        self.pctl = float(pctl)

        self.encoder = _Encoder(img_size=self.img_size)
        self.knn: Optional[NearestNeighbors] = None
        self.threshold: float = float("inf")  # fallback: hepsi normal kabul

        loaded = self._try_load_artifacts()
        if not loaded:
            self._try_fit_from_data()

    def _try_load_artifacts(self) -> bool:
        meta = self.artifacts_dir / "meta.json"
        feats = self.artifacts_dir / "train_feats.npy"
        if not (meta.exists() and feats.exists()):
            return False
        try:
            meta_obj = json.loads(meta.read_text(encoding="utf-8"))
            self.k = int(meta_obj.get("k", self.k))
            self.pctl = float(meta_obj.get("pctl", self.pctl))
            self.threshold = float(meta_obj.get("threshold", float("inf")))
            feats_np = np.load(feats)
            self.knn = NearestNeighbors(n_neighbors=self.k, metric="euclidean").fit(feats_np)
            return True
        except Exception:
            return False

    def _try_fit_from_data(self) -> bool:
        if self.data_dir is None:
            return False
        ok_dir = self.data_dir / "train" / "good"
        paths = _list_images(ok_dir)
        if len(paths) == 0:
            return False
        feats = []
        B = 32
        for i in range(0, len(paths), B):
            pil = [Image.open(p).convert("RGB") for p in paths[i:i+B]]
            feats.append(self.encoder.encode_batch(pil))
        feats = np.concatenate(feats, axis=0)

        knn_train = NearestNeighbors(n_neighbors=self.k + 1, metric="euclidean").fit(feats)
        dists, _ = knn_train.kneighbors(feats)
        scores = dists[:, 1:].mean(axis=1)
        self.threshold = float(np.percentile(scores, self.pctl))

        self.knn = NearestNeighbors(n_neighbors=self.k, metric="euclidean").fit(feats)
        return True

    def score_pil(self, pil_img: Image.Image) -> float:
        if self.knn is None:
            return 0.0
        z = self.encoder.encode_pil(pil_img)
        dists, _ = self.knn.kneighbors(z)
        return float(dists.mean())

    def score_bbox(self, frame_bgr: np.ndarray, xyxy: Tuple[int,int,int,int], pad_ratio: float = 0.05) -> float:
        x1, y1, x2, y2 = map(int, xyxy)
        H, W = frame_bgr.shape[:2]
        w = max(1, x2 - x1); h = max(1, y2 - y1)
        px = int(pad_ratio * w); py = int(pad_ratio * h)
        xa = max(0, x1 - px); ya = max(0, y1 - py)
        xb = min(W, x2 + px); yb = min(H, y2 + py)
        crop = frame_bgr[ya:yb, xa:xb]
        if crop.size == 0:
            return 0.0
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        return self.score_pil(pil)

    def is_anomaly(self, score: float) -> bool:
        return score >= self.threshold
