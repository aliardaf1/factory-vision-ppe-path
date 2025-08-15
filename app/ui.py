# app/ui.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import time
import json
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import torch
from shapely.geometry import Point, Polygon
import supervision as sv

from detectors.yolo_v8 import YOLOv8Detector
from core.region_timer import RegionStayTimer

# ==== AYARLAR ====
VIOL_THRESH = 1.0           # yol DIŞI kalma eşiği (saniye)
CAM_INDEX   = 0             # kamera indexi
VIEW_W, VIEW_H = 1280, 720  # ekranda gösterilecek sabit boyut

MODEL_PATH_DEFAULT = "yolov8n.pt"           # COCO ile başla; PPE modeli gelince değiştir
POLY_JSON_DEFAULT  = "app/path_polygon.json"

# =================

class FactoryVisionApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Factory Vision — PPE & Path Monitor")
        self.root.configure(bg="black")
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda e: self._on_exit())

        # --- ÜST SABİT PANEL ---
        top = tk.Frame(self.root, bg="black")
        top.pack(side="top", fill="x", pady=8)

        btn_style = dict(bg="gray20", fg="white", font=("Arial", 14), relief="flat", padx=16, pady=6)
        tk.Button(top, text="Başlat", command=self.start, **btn_style).pack(side="left", padx=10)
        tk.Button(top, text="Durdur", command=self.stop,  **btn_style).pack(side="left", padx=10)

        # seçenekler
        self.chk_ppe = tk.BooleanVar(value=True)
        self.chk_path = tk.BooleanVar(value=True)
        opt_style = dict(bg="black", fg="white", selectcolor="black", activebackground="black",
                         activeforeground="white", font=("Arial", 12))
        tk.Checkbutton(top, text="PPE (Baret/Yelek)", variable=self.chk_ppe, **opt_style).pack(side="left", padx=20)
        tk.Checkbutton(top, text="Sarı Yol Dışı", variable=self.chk_path, **opt_style).pack(side="left")

        # durum yazıları
        self.fps_lbl = tk.Label(top, text="FPS: -", bg="black", fg="white", font=("Arial", 14))
        self.fps_lbl.pack(side="right", padx=20)
        self.status_lbl = tk.Label(top, text="Durum: Hazır", bg="black", fg="white", font=("Arial", 12))
        self.status_lbl.pack(side="right", padx=20)

        # --- ORTA KAMERA ALANI (SABİT) ---
        mid = tk.Frame(self.root, bg="black")
        mid.pack(expand=True)
        self.video_label = tk.Label(mid, bg="black")
        self.video_label.pack()

        # --- ALT SABİT PANEL (dosya yolları) ---
        bottom = tk.Frame(self.root, bg="black")
        bottom.pack(side="bottom", fill="x", pady=8)

        lbl_style = dict(bg="black", fg="white", font=("Arial", 11))
        ent_style = dict(bg="gray10", fg="white", insertbackground="white", relief="flat")

        tk.Label(bottom, text="Model:", **lbl_style).pack(side="left", padx=(12,5))
        self.model_path_var = tk.StringVar(value=MODEL_PATH_DEFAULT)
        tk.Entry(bottom, textvariable=self.model_path_var, width=36, **ent_style).pack(side="left")

        tk.Label(bottom, text="Yol Poligonu:", **lbl_style).pack(side="left", padx=(20,5))
        self.poly_path_var = tk.StringVar(value=POLY_JSON_DEFAULT)
        tk.Entry(bottom, textvariable=self.poly_path_var, width=36, **ent_style).pack(side="left")

        # --- ARKA PLAN DURUMLARI ---
        self.cap = None
        self.running = False
        self.last_tick = time.time()
        self.fps = 0.0

        # model / tracker / timer / poligon
        self.detector = None
        self.class_names = {}
        self.tracker = sv.ByteTrack()
        self.timer = RegionStayTimer()
        self.path_poly = None

        # model ve poligonu ön-yükle (hata verirse yine de UI açılır)
        self._ensure_model()
        self._load_polygon()

    # ===== Yardımcılar =====
    def _ensure_model(self):
        if self.detector is not None:
            return
        try:
            self.detector = YOLOv8Detector()
            self.detector.load(self.model_path_var.get().strip() or MODEL_PATH_DEFAULT, device="cuda")
            self.class_names = self.detector.class_names()
            self._set_status(f"Model yüklendi: {os.path.basename(self.model_path_var.get()) or MODEL_PATH_DEFAULT}")
        except Exception as e:
            self._set_status(f"Model yüklenemedi: {e}")

    def _load_polygon(self):
        path = self.poly_path_var.get().strip() or POLY_JSON_DEFAULT
        try:
            with open(path, "r", encoding="utf-8") as f:
                coords = json.load(f)
            if not (isinstance(coords, list) and len(coords) >= 3):
                raise ValueError("Geçersiz poligon")
            self.path_poly = Polygon(coords)
            self._set_status(f"Poligon yüklendi ({len(coords)} nokta)")
        except Exception as e:
            self.path_poly = None
            self._set_status(f"Poligon yok: {e}")

    def _set_status(self, msg: str):
        self.status_lbl.config(text=f"Durum: {msg}")

    def _on_exit(self):
        self.stop()
        self.root.destroy()

    # ===== Kamera Kontrolleri =====
    def start(self):
        if self.running:
            return
        self._ensure_model()
        self._load_polygon()
        self.cap = cv2.VideoCapture(CAM_INDEX)
        if not self.cap.isOpened():
            self._set_status("Kamera açılamadı")
            return
        self.running = True
        self._set_status("Çalışıyor")
        self.last_tick = time.time()
        self.update_frame()

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._set_status("Durduruldu")

    # ===== Ana Döngü =====
    def update_frame(self):
        if not self.running:
            return

        ok, frame = self.cap.read()
        if not ok:
            self.stop()
            return

        # sabit gösterim boyutuna ölçekle
        frame = cv2.resize(frame, (VIEW_W, VIEW_H), interpolation=cv2.INTER_AREA)

        # 1) YOLO tespit
        dets = self.detector.predict(frame, conf=0.35, imgsz=640)
        out = frame.copy()

        # 2) Yol poligonunu çiz
        if self.path_poly is not None:
            pts = np.array(self.path_poly.exterior.coords, dtype=np.int32)
            cv2.polylines(out, [pts], True, (0, 255, 255), 2)

        # 3) Sınıf id listeleri
        names = self.class_names
        person_ids = [i for i, n in names.items() if str(n).lower() in ("person", "pedestrian")]
        helmet_ids = [i for i, n in names.items() if "helmet" in str(n).lower()]
        vest_ids   = [i for i, n in names.items() if "vest"   in str(n).lower()]

        # Yardımcılar
        def center(box):
            x1, y1, x2, y2 = box
            return (int((x1 + x2) / 2), int((y1 + y2) / 2))

        def inside(box_a, pt):
            x1, y1, x2, y2 = box_a
            return (x1 <= pt[0] <= x2) and (y1 <= pt[1] <= y2)

        def foot(box):
            x1, y1, x2, y2 = box
            return (int((x1 + x2) / 2), int(y2))

        # 4) Tespitleri ayır
        persons = [d for d in dets if d.cls_id in person_ids]
        helmets = [d for d in dets if d.cls_id in helmet_ids]
        vests   = [d for d in dets if d.cls_id in vest_ids]

        # 5) Tracking (yalnız person)
        if persons:
            xyxy  = np.array([p.box for p in persons], dtype=np.float32)
            confs = np.array([p.score for p in persons], dtype=np.float32)
            cids  = np.array([p.cls_id for p in persons], dtype=np.int32)
            det_sv = sv.Detections(xyxy=xyxy, confidence=confs, class_id=cids)
            tracks = self.tracker.update_with_detections(det_sv)
        else:
            tracks = sv.Detections.empty()

        # 6) Her track için kurallar
        for i in range(len(tracks)):
            x1, y1, x2, y2 = map(int, tracks.xyxy[i])
            track_id = int(tracks.tracker_id[i]) if tracks.tracker_id is not None else i
            p_box  = (x1, y1, x2, y2)
            p_foot = foot(p_box)

            # PPE
            has_h = any(inside(p_box, center(h.box)) for h in helmets) if self.chk_ppe.get() else True
            has_v = any(inside(p_box, center(v.box)) for v in vests)   if self.chk_ppe.get() else True

            # Yol
            in_path = True
            if self.chk_path.get() and self.path_poly is not None:
                in_path = self.path_poly.contains(Point(*p_foot))

            # Süre bazlı ihlal
            violation_time = self.timer.update(track_id, condition=(not in_path))
            violation = (violation_time >= VIOL_THRESH)

            ok_person = (in_path and has_h and has_v and not violation)
            color = (0, 200, 0) if ok_person else (0, 0, 255)

            parts = [f"ID:{track_id}"]
            if self.chk_path.get():
                parts += [f"Iceride:{int(in_path)}", f"Saniye:{violation_time:.1f}s", f"Ihlal:{int(violation)}"]
            if self.chk_ppe.get():
                parts += [f"Baret:{int(has_h)}", f"Yelek:{int(has_v)}"]
            label = " ".join(parts)

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(out, p_foot, 4, (255, 255, 255), -1)

        # 7) FPS
        now = time.time()
        dt = now - self.last_tick
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps > 0 else (1.0 / dt)
        self.last_tick = now
        self.fps_lbl.config(text=f"FPS: {self.fps:.1f}")

        # 8) TK görüntüle (siyah arkaplanda)
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # 9) Döngü
        self.root.after(30, self.update_frame)

    # ===== Çalıştır =====
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = FactoryVisionApp()
    if torch.cuda.is_available():
        print("GPU aktif:", torch.cuda.get_device_name(0))
    else:
        print("GPU kullanılmıyor, CPU'da çalışıyor")
    app.run()
