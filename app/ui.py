# app/ui.py
import sys, os, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from app.pipeline import ModelManager, YoloDetAdapter, YoloSegAdapter

# ================== AYARLAR ==================
VIEW_W, VIEW_H = 1280, 720  # görüntüyü buna göre ölçekleriz
# Modeller: asıl (varsa) + fallback (kesin yerel)
PPE_DET_WEIGHTS   = r"models/ppe_det.pt"
PATH_SEG_WEIGHTS  = r"models/line_detection.pt"
FALLBACK_DET      = r"yolov8n.pt"
FALLBACK_SEG      = r"yolov8n-seg.pt"
# =============================================


class FactoryVisionApp:
    def __init__(self):
        # ---- PENCERE / TEMA ----
        self.root = tk.Tk()
        self.root.title("Factory Vision — Kaynak Seçimli UI")
        self.root.configure(bg="black")
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda e: self._on_exit())

        # ---- DURUMLAR ----
        self.source_type = tk.StringVar(value="Webcam")  # Webcam | Resim | Video
        self.source_path = None
        self.cap = None
        self.running = False
        self.frame_idx = 0
        self.last_tick = time.time()
        self.fps = 0.0

        # ---- SOL KONTROL PANELİ ----
        left = tk.Frame(self.root, bg="black")
        left.pack(side="left", fill="y", padx=16, pady=16)

        lbl_style = dict(bg="black", fg="white", font=("Arial", 12))
        btn_style = dict(bg="gray20", fg="white", relief="flat", font=("Arial", 12), padx=10, pady=6)

        tk.Label(left, text="Kaynak:", **lbl_style).pack(anchor="w")
        self.combo_source = ttk.Combobox(
            left, values=["Webcam", "Resim", "Video"],
            textvariable=self.source_type, state="readonly", width=18
        )
        self.combo_source.pack(anchor="w", pady=6)
        self.combo_source.bind("<<ComboboxSelected>>", self.on_source_change)

        self.btn_select = tk.Button(left, text="Dosya Seç", command=self.select_file, **btn_style)
        self.btn_select.pack(anchor="w", pady=6, fill="x")
        self.btn_select.config(state="disabled")

        # Başlat / Durdur
        self.btn_start = tk.Button(left, text="Başlat", command=self.start, **btn_style)
        self.btn_start.pack(anchor="w", pady=(16, 6), fill="x")
        self.btn_stop = tk.Button(left, text="Durdur", command=self.stop, **btn_style)
        self.btn_stop.pack(anchor="w", pady=6, fill="x")

                # ------------------ MODEL CHECKBOXLARI ------------------
        tk.Label(left, text="Aktif Modeller:", **lbl_style).pack(anchor="w", pady=(16, 4))
        self.var_ppe = tk.BooleanVar(value=True)
        self.var_path = tk.BooleanVar(value=True)
        tk.Checkbutton(left, text="PPE Detection", variable=self.var_ppe,
                       bg="black", fg="white", selectcolor="black",
                       activebackground="black", command=self.update_model_flags).pack(anchor="w")
        tk.Checkbutton(left, text="Path Segmentation", variable=self.var_path,
                       bg="black", fg="white", selectcolor="black",
                       activebackground="black", command=self.update_model_flags).pack(anchor="w")
        
        # FPS ve durum
        self.fps_lbl = tk.Label(left, text="FPS: -", **lbl_style)
        self.fps_lbl.pack(anchor="w", pady=(16, 4))
        self.status_lbl = tk.Label(left, text="Durum: Hazır", **lbl_style)
        self.status_lbl.pack(anchor="w")

        # ---- SAĞ GÖRÜNTÜ PANELİ ----
        right = tk.Frame(self.root, bg="black")
        right.pack(side="right", expand=True, fill="both")
        self.video_label = tk.Label(right, bg="black")
        self.video_label.pack(expand=True)

        # ---- MODEL MANAGER ----
        self.mm = ModelManager()
        self.mm.register(
            "path",
            YoloSegAdapter(PATH_SEG_WEIGHTS, name_prefix="path",
                           conf=0.1, imgsz=640, interval=1,
                           fallback_path=FALLBACK_SEG, allow_download=False),
            enabled=True
        )
        self.mm.register(
            "ppe",
            YoloDetAdapter(PPE_DET_WEIGHTS, name_prefix="ppe",
                           conf=0.40, imgsz=640, interval=1,
                           fallback_path=FALLBACK_DET, allow_download=False),
            enabled=True
        )

    # ------------------------- Yardımcılar -------------------------
    def _set_status(self, text: str):
        self.status_lbl.config(text=f"Durum: {text}")

    def _on_exit(self):
        self.stop()
        self.root.destroy()
    
    # ------------------ Model checkbox güncelle ------------------
    def update_model_flags(self):
        self.mm.set_enabled("ppe", self.var_ppe.get())
        self.mm.set_enabled("path", self.var_path.get())


    # ------------------------- Kaynak seçim & dosya diyalogu -------------------------
    def on_source_change(self, event=None):
        sel = self.source_type.get()
        if sel in ("Resim", "Video"):
            self.btn_select.config(state="normal")
        else:
            self.btn_select.config(state="disabled")
            self.source_path = None

    def select_file(self):
        sel = self.source_type.get()
        if sel == "Resim":
            path = filedialog.askopenfilename(
                title="Görüntü Seç", filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All Files", "*.*")]
            )
        elif sel == "Video":
            path = filedialog.askopenfilename(
                title="Video Seç", filetypes=[("Videos", "*.mp4 *.avi *.mkv *.mov *.wmv"), ("All Files", "*.*")]
            )
        else:
            return
        if path:
            self.source_path = path
            self._set_status(f"Seçildi: {os.path.basename(path)}")

    # ------------------------- Başlat / Durdur -------------------------
    def start(self):
        if self.running:
            return
        sel = self.source_type.get()

        if sel == "Webcam":
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Hata", "Kamera açılamadı.")
                return
        elif sel == "Video":
            if not self.source_path:
                messagebox.showwarning("Uyarı", "Önce bir video dosyası seçin.")
                return
            self.cap = cv2.VideoCapture(self.source_path)
            if not self.cap.isOpened():
                messagebox.showerror("Hata", "Video açılamadı.")
                return
        elif sel == "Resim":
            if not self.source_path:
                messagebox.showwarning("Uyarı", "Önce bir resim seçin.")
                return
            self.cap = None  # tek kare için capture gerekmez
        else:
            return

        self.running = True
        self.frame_idx = 0
        self.last_tick = time.time()
        self._set_status("Çalışıyor")
        self.update_frame()

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._set_status("Durduruldu")

    # ------------------------- Görüntü işleme döngüsü -------------------------
    def update_frame(self):
        if not self.running:
            return

        # 1) Kaynaktan frame al
        frame = None
        sel = self.source_type.get()

        if sel == "Resim":
            frame = cv2.imread(self.source_path)
            if frame is None:
                self._set_status("Resim okunamadı!")
                self.running = False
        elif self.cap is not None:
            ok, frm = self.cap.read()
            if not ok:
                self.stop()
                return
            frame = frm

        if frame is None:
            return

        # 2) Ölçekle (gösterim için)
        frame = self._resize_keep_aspect(frame, (VIEW_W, VIEW_H))
        out = frame.copy()

        # 3) Inference (çoklu model)
        res = self.mm.infer(frame, frame_idx=self.frame_idx)
        self.frame_idx += 1
        
        # Debug: Mask bilgilerini yazdır
        print(f"Frame {self.frame_idx}: Found {len(res['masks'])} masks")
        for m in res['masks']:
            print(f"Mask class: {m.cls_name}, source: {m.source}")

        # 4) Mask overlay (walk_zone)
        # path:* sınıflarını tek maske gibi uygula (ilk bulunanı alıyoruz)
        path_mask = None
        for m in res["masks"]:
            if m.cls_name.lower().startswith("path:"):
                # Maskeyi görüntü boyutuna uygun şekilde yeniden boyutlandır
                mask = m.mask.astype(bool)
                h, w = out.shape[:2]
                path_mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                break
        
        if path_mask is not None:
            color = np.array((0, 180, 80), dtype=np.float32)  # yeşilimsi
            out[path_mask] = (0.45 * color + 0.55 * out[path_mask]).astype(np.uint8)

        # 5) BBox çiz
        for d in res["detections"]:
            x1, y1, x2, y2 = map(int, d.xyxy)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 2)
            label = f"{d.cls_name} {d.conf:.2f}"
            cv2.putText(out, label, (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        # 6) FPS
        now = time.time()
        dt = now - self.last_tick
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps > 0 else (1.0 / dt)
        self.last_tick = now
        self.fps_lbl.config(text=f"FPS: {self.fps:.1f}")

        # 7) TK gösterim
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Resim tek kare olduğu için otomatik durdur
        if sel == "Resim":
            self.running = False
            self._set_status("Tek kare gösterildi.")
            return

        # Video/Webcam için döngü
        if self.running:
            self.root.after(1, self.update_frame)

    # ------------------------- Yardımcı: oranı koruyarak boyutlandır -------------------------
    @staticmethod
    def _resize_keep_aspect(img, target_hw):
        tw, th = target_hw
        h, w = img.shape[:2]
        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)
        if nw <= 0 or nh <= 0:
            return cv2.resize(img, (tw, th))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        # siyah zemin üzerine ortala
        canvas = np.zeros((th, tw, 3), dtype=np.uint8)
        y0 = (th - nh) // 2
        x0 = (tw - nw) // 2
        canvas[y0:y0+nh, x0:x0+nw] = resized
        return canvas

    # ------------------------- Çalıştır -------------------------
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    FactoryVisionApp().run()
