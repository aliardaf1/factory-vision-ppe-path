# app/ui.py
import sys, os, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from logger import ppe_episode_tracker
from app.pipeline import ModelManager, YoloDetAdapter, YoloSegAdapter
import json
from datetime import datetime

# ================== AYARLAR ==================
VIEW_W, VIEW_H = 1280, 720  # görüntüyü buna göre ölçekleriz
# Modeller: asıl (varsa) + fallback (kesin yerel)
PPE_DET_WEIGHTS   = r"models/ppe_model_v1.pt"
PATH_SEG_WEIGHTS  = r"models/line_detection.pt"
FALLBACK_DET      = r"yolov8n.pt"
FALLBACK_SEG      = r"yolov8n-seg.pt"
# =============================================
ALIAS_MAP = {
    # helmet (+)
    "helmet": "helmet",
    "hardhat": "helmet",
    "head_helmet": "helmet",

    # helmet (-)
    "not_helmet": "not_helmet",
    "no_hardhat": "not_helmet",
    "nohelmet": "not_helmet",
    "head_nohelmet": "not_helmet",

    # vest (+)
    "vest": "vest",
    "vests": "vest",
    "safety_vest": "vest",
    "safetyvest": "vest",
    "reflective_vest": "vest",
    "reflectivevest": "vest",
    "reflective": "vest",           # ← özellikle istedin

    # vest (-)
    "not_vest": "not_vest",
    "no_safety_vest": "not_vest",
    "no_safetyvest": "not_vest",
    "not_reflective": "not_vest",

    # ilgisizleri at (None verirsen filtre aşağıda zaten skip eder)
    "no_mask": None,
    "nomask": None,
}

CANONICAL = {"helmet", "vest", "not_helmet", "not_vest"}
DEFAULT_PCC = {"helmet": 0.70, "vest": 0.70, "not_helmet": 0.50, "not_vest": 0.50}

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
                           conf=0.20, imgsz=768, interval=1,
                           fallback_path=FALLBACK_DET, allow_download=False,
                           alias_map=ALIAS_MAP,              # alias aktif
                           include_canonical=CANONICAL,      # sadece 4 sınıf
                           per_class_conf=DEFAULT_PCC,        # sınıf başı eşik),
                           )
        )
        # ---- PPE EPISODE TRACKER ----
        self.tracker = ppe_episode_tracker.PPEEpisodeTracker(site="warehouse-A", camera_id="cam-01",
                                 mode="global", also_console=True,
                                 source_kind="video", evidence_delay_s=5.0)

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

        try:
            self.tracker.flush()
        except Exception:
            pass

    def update_frame(self):
        if not self.running:
            return

        # ---- 0) Ayarlar / cache ----
        DETECT_EVERY = getattr(self, "DETECT_EVERY", 2)   # her N karede bir inference
        if not hasattr(self, "_last_res"):
            self._last_res = {"detections": [], "masks": []}

        # ---- 1) Kaynaktan HAM kareyi al ----
        sel = self.source_type.get()  # "Resim" | "Video" | "Webcam" vb.
        frame_raw = None

        if sel == "Resim":
            # Fotoğrafta periyodik log için döngüyü DURDURMA!
            if not hasattr(self, "_static_image"):
                img = cv2.imread(self.source_path)
                if img is None:
                    self._set_status("Resim okunamadı!")
                    self.running = False
                    return
                self._static_image = img
            frame_raw = self._static_image.copy()
        elif self.cap is not None:
            ok, frm = self.cap.read()
            if not ok:
                self.stop()
                return
            frame_raw = frm

        if frame_raw is None:
            return

        H0, W0 = frame_raw.shape[:2]

        # ---- 2) Inference: her N karede bir, arada son çıktıyı kullan ----
        cur_idx = self.frame_idx
        do_infer = (cur_idx % max(1, DETECT_EVERY) == 0)
        if do_infer:
            res = self.mm.infer(frame_raw, frame_idx=cur_idx)
            self._last_res = res
        else:
            res = self._last_res
        self.frame_idx += 1

        # ---- 3) LOGGING: 4 kanonik sınıfa göre missing/conf çıkar ve tracker'a gönder ----
        def _canon(n: str) -> str:
            n = str(n).split(":", 1)[-1].lower().replace("-", " ")
            n = "_".join(n.split())  # "NO Safety Vest" -> "no_safety_vest"
            # alias → kanonik
            if n in {"helmet", "hardhat", "head_helmet"}:
                return "helmet"
            if n in {"vest", "vests", "safety_vest", "reflective_vest", "reflective"}:
                return "vest"
            if n in {"not_helmet", "no_hardhat", "nohelmet", "head_nohelmet"}:
                return "not_helmet"
            if n in {"not_vest", "no_safety_vest", "no_safetyvest"}:
                return "not_vest"
            if n == "not_reflective" or "mask" in n:
                return ""  # ignore
            return n

        # frame içi max skorları topla
        max_conf_frame = {"helmet": 0.0, "vest": 0.0, "not_helmet": 0.0, "not_vest": 0.0}
        for d in res["detections"]:
            name = _canon(d.cls_name)
            if name in max_conf_frame:
                c = float(d.conf)
                if c > max_conf_frame[name]:
                    max_conf_frame[name] = c

        # eksikler: pozitif yoksa veya not_* varsa
        missing = []
        if (max_conf_frame["helmet"] <= 0.0) or (max_conf_frame["not_helmet"] > 0.0):
            missing.append("helmet")
        if (max_conf_frame["vest"]   <= 0.0) or (max_conf_frame["not_vest"]   > 0.0):
            missing.append("vest")

        try:
            self.tracker.process_frame(
                frame_id=int(cur_idx),
                frame_ts=time.time(),
                observations=[{
                    "person_id": None,
                    "missing": missing,
                    "conf": {k: v for k, v in max_conf_frame.items() if v > 0.0}
                }]
            )
        except Exception:
            # logger'da bir sorun olsa bile UI aksın
            pass

        # ---- 4) ÇİZİM: ham boyutta çiz, sonda tek kez küçült ----
        out_raw = frame_raw.copy()

        # 4.a) Yürüyüş yolu maskesi (varsa) ham boyuta eşle
        path_mask = None
        for m in res["masks"]:
            if m.cls_name.lower().startswith("path:"):
                mask = m.mask.astype(bool)
                mh, mw = mask.shape[:2]
                if (mh, mw) != (H0, W0):
                    mask = cv2.resize(mask.astype(np.uint8), (W0, H0),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
                path_mask = mask
                break
        if path_mask is not None:
            color = np.array((0, 180, 80), dtype=np.float32)
            out_raw[path_mask] = (0.45 * color + 0.55 * out_raw[path_mask]).astype(np.uint8)

        # 4.b) Sadece {helmet, vest, not_helmet, not_vest} kutularını çiz
        for d in res["detections"]:
            name = _canon(d.cls_name)
            if name not in {"helmet", "vest", "not_helmet", "not_vest"}:
                continue
            x1, y1, x2, y2 = map(int, d.xyxy)
            color = (0, 200, 0) if name in {"helmet", "vest"} else (0, 0, 255)
            cv2.rectangle(out_raw, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out_raw, f"{d.cls_name} {d.conf:.2f}",
                        (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ---- 5) Gösterim için küçült ----
        out = self._resize_keep_aspect(out_raw, (VIEW_W, VIEW_H))

        # ---- 6) FPS ----
        now = time.time()
        dt = now - self.last_tick
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps > 0 else (1.0 / dt)
        self.last_tick = now
        self.fps_lbl.config(text=f"FPS: {self.fps:.1f}")

        # ---- 7) TK gösterim ----
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # ---- 8) Döngü ----
        # Fotoğrafta da döngüye devam (5 sn aralıklı loglar için); kullanıcı stop ile durdurur.
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
