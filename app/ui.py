# app/ui.py
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2, json, time, os, sys
import numpy as np

# proje kökünü modül yoluna ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detectors.yolo_v8 import YOLOv8Detector
from core.region_timer import RegionStayTimer
from shapely.geometry import Point, Polygon

VIOL_THRESH = 1.0  # yol DIŞI kalma eşiği (saniye)

class FactoryVisionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Factory Vision - PPE & Path Monitor")
        self.root.geometry("1100x680")

        # durum
        self.cap = None
        self.running = False
        self.detector = None
        self.class_names = {}
        self.timer = RegionStayTimer()
        self.path_poly = None  # Polygon ya da None
        self.last_tick = time.time()
        self.fps = 0.0

        # Sol panel
        self.left = ttk.Frame(self.root, padding=10)
        self.left.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(self.left, text="Aktif Denetlemeler", font=("Arial", 12, "bold")).pack(pady=6, anchor="w")

        self.chk_ppe_var = tk.BooleanVar(value=True)
        self.chk_path_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(self.left, text="Baret / Yelek Tespiti", variable=self.chk_ppe_var).pack(anchor="w", pady=2)
        ttk.Checkbutton(self.left, text="Sarı Yol Dışı Tespiti", variable=self.chk_path_var).pack(anchor="w", pady=2)

        ttk.Separator(self.left, orient="horizontal").pack(fill="x", pady=8)

        # model ve poligon seçim alanı
        ttk.Label(self.left, text="Model Ağırlığı (.pt):").pack(anchor="w")
        self.model_path = tk.StringVar(value="yolov8n.pt")  # COCO ile başla; sonra models/ppe_yolov8.pt
        self.model_entry = ttk.Entry(self.left, textvariable=self.model_path, width=35)
        self.model_entry.pack(anchor="w", pady=2)

        ttk.Label(self.left, text="Yol Poligonu (JSON):").pack(anchor="w")
        self.poly_path = tk.StringVar(value="app/path_polygon.json")
        self.poly_entry = ttk.Entry(self.left, textvariable=self.poly_path, width=35)
        self.poly_entry.pack(anchor="w", pady=2)

        self.start_btn = ttk.Button(self.left, text="Başlat", command=self.start_camera)
        self.start_btn.pack(fill=tk.X, pady=6)
        self.stop_btn = ttk.Button(self.left, text="Durdur", command=self.stop_camera, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X)

        ttk.Separator(self.left, orient="horizontal").pack(fill="x", pady=8)

        self.status_lbl = ttk.Label(self.left, text="Durum: Hazır")
        self.status_lbl.pack(anchor="w", pady=4)

        self.fps_lbl = ttk.Label(self.left, text="FPS: -")
        self.fps_lbl.pack(anchor="w", pady=2)

        # görüntü paneli
        self.video_label = ttk.Label(self.root)
        self.video_label.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

    # --- yardımcılar ---
    def _load_polygon(self, path_json: str):
        try:
            with open(path_json, "r", encoding="utf-8") as f:
                coords = json.load(f)
            if not (isinstance(coords, list) and len(coords) >= 3):
                raise ValueError("Geçersiz poligon")
            self.path_poly = Polygon(coords)
            self.status_lbl.config(text=f"Durum: Poligon yüklendi ({len(coords)} nokta)")
            return True
        except Exception as e:
            self.path_poly = None
            self.status_lbl.config(text=f"Durum: Poligon YOK ({e})")
            return False

    def _ensure_detector(self):
        if self.detector is None:
            self.detector = YOLOv8Detector()
            weights = self.model_path.get().strip()
            self.detector.load(weights)
            self.class_names = self.detector.class_names()
            self.status_lbl.config(text=f"Durum: Model yüklendi ({os.path.basename(weights)})")

    # --- UI aksiyonları ---
    def start_camera(self):
        # model ve poligonları yükle
        try:
            self._ensure_detector()
        except Exception as e:
            messagebox.showerror("Model Yükleme Hatası", str(e))
            return

        self._load_polygon(self.poly_path.get().strip())

        # kamera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Kamera", "Kamera açılamadı.")
            return

        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_lbl.config(text="Durum: Çalışıyor")
        self.last_tick = time.time()
        self.update_frame()

    def stop_camera(self):
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        if self.cap:
            self.cap.release()
        self.status_lbl.config(text="Durum: Durduruldu")

    # --- ana döngü (Tkinter after) ---
    def update_frame(self):
        if not self.running:
            return

        ok, frame = self.cap.read()
        if not ok:
            self.stop_camera()
            return

        # inference
        dets = self.detector.predict(frame, conf=0.35, imgsz=640)

        out = frame.copy()

        # Sarı yolu çiz (varsa)
        if self.path_poly is not None:
            pts = np.array(self.path_poly.exterior.coords, dtype=np.int32)
            cv2.polylines(out, [pts], True, (0, 255, 255), 2)

        # sınıf eşleme
        names = self.class_names
        person_ids = [i for i, n in names.items() if str(n).lower() in ("person", "pedestrian")]
        helmet_ids = [i for i, n in names.items() if "helmet" in str(n).lower()]
        vest_ids   = [i for i, n in names.items() if "vest"   in str(n).lower()]

        # basit PPE eşleme: helmet/vest merkezleri person kutusu içinde mi
        def center(box):
            x1, y1, x2, y2 = box
            return (int((x1 + x2) / 2), int((y1 + y2) / 2))
        def inside(box_a, pt):
            x1, y1, x2, y2 = box_a
            return (x1 <= pt[0] <= x2) and (y1 <= pt[1] <= y2)
        def foot(box):
            x1, y1, x2, y2 = box
            return (int((x1 + x2) / 2), int(y2))

        persons = [d for d in dets if d.cls_id in person_ids]
        helmets = [d for d in dets if d.cls_id in helmet_ids]
        vests   = [d for d in dets if d.cls_id in vest_ids]

        # Yol dışı kontrol: tracking yoksa, basit “anlık” gösterim (timer, ileride tracking ekleyince tam çalışacak)
        # Şimdilik RegionStayTimer'ı bbox merkezine ID veremediğimiz için kullanmıyoruz.
        # Tracking eklediğinde aynı ID için self.timer.update(track_id, not in_path) çağır.
        for p in persons:
            x1, y1, x2, y2 = p.box
            p_center = center(p.box)
            p_foot   = foot(p.box)

            # PPE kontrolü (isteğe bağlı)
            has_h = any(inside(p.box, center(h.box)) for h in helmets) if self.chk_ppe_var.get() else True
            has_v = any(inside(p.box, center(v.box)) for v in vests)   if self.chk_ppe_var.get() else True

            # Yol kontrolü
            in_path = True
            if self.chk_path_var.get() and self.path_poly is not None:
                in_path = self.path_poly.contains(Point(p_foot[0], p_foot[1]))

            ok_person = in_path and has_h and has_v
            color = (0, 200, 0) if ok_person else (0, 0, 255)

            msg = []
            if self.chk_path_var.get():
                msg.append(f"in:{int(in_path)}")
            if self.chk_ppe_var.get():
                msg.append(f"H:{int(has_h)} V:{int(has_v)}")

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            if msg:
                cv2.putText(out, " ".join(msg), (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(out, p_foot, 4, (255, 255, 255), -1)

        # FPS
        now = time.time()
        dt = now - self.last_tick
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps > 0 else (1.0 / dt)
        self.last_tick = now
        self.fps_lbl.config(text=f"FPS: {self.fps:.1f}")

        # TK görüntüle
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # sonraki kare
        self.root.after(30, self.update_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = FactoryVisionUI(root)
    root.mainloop()
