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
from logger import fire_episode_logger

# ================== AYARLAR ==================
VIEW_W, VIEW_H = 1280, 720  # görüntüyü buna göre ölçekleriz
# Modeller: asıl (varsa) + fallback (kesin yerel)
PPE_DET_WEIGHTS   = r"models/ppe_model_v1.pt"
FIRE_DET_WEIGHTS  = r"models/fire_detection.pt"
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

    # ilgisizleri at
    "no_mask": None,
    "nomask": None,
}


CANONICAL = {"helmet", "vest", "not_helmet", "not_vest"}
DEFAULT_PCC = {"helmet": 0.70, "vest": 0.70, "not_helmet": 0.50, "not_vest": 0.50}

FIRE_ALIAS = {
    "fire": "fire", "Fire:": "fire", "flame": "fire",
    "smoke": "smoke", "Smoke": "smoke"
}
FIRE_CANON = {"fire", "smoke"}

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
        self.var_fire = tk.BooleanVar(value=True)
        tk.Checkbutton(left, text="PPE Detection", variable=self.var_ppe,
                       bg="black", fg="white", selectcolor="black",
                       activebackground="black", command=self.update_model_flags).pack(anchor="w")
        tk.Checkbutton(left, text="Path Segmentation", variable=self.var_path,
                       bg="black", fg="white", selectcolor="black",
                       activebackground="black", command=self.update_model_flags).pack(anchor="w")
        tk.Checkbutton(left, text="Fire Detection", variable=self.var_fire,
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
            enabled=self.var_path.get()
        )
        self.mm.register(
            "ppe",
            YoloDetAdapter(PPE_DET_WEIGHTS, name_prefix="ppe",
                           conf=0.20, imgsz=768, interval=1,
                           fallback_path=FALLBACK_DET, allow_download=False,
                           alias_map=ALIAS_MAP,              # alias aktif
                           include_canonical=CANONICAL,      # sadece 4 sınıf
                           per_class_conf=DEFAULT_PCC        # sınıf başı eşik
                           ),
            enabled=self.var_ppe.get()
        )
        self.mm.register(
            "fire",
            YoloDetAdapter(
                FIRE_DET_WEIGHTS, name_prefix="fire",
                conf=0.325, imgsz=640, interval=1,
                fallback_path=FALLBACK_DET, allow_download=False,
                alias_map=FIRE_ALIAS,             # ← eklendi
                include_canonical=FIRE_CANON,      # ← eklendi
            ),
            enabled=self.var_fire.get()
        )

        # ---- PPE EPISODE TRACKER ----
        self.tracker = ppe_episode_tracker.PPEEpisodeTracker(
            site="warehouse-A", camera_id="cam-01",
            mode="global", also_console=True,
            source_kind="video",        # varsayılan: video (kaynak seçimi ile senkronlayacağız)
            evidence_delay_s=5.0
        )

        # Fire logger
        self.fire_logger = fire_episode_logger.FireEpisodeTracker(
            site="warehouse-B", camera_id="cam-02",
            source_kind="video",          # on_source_change ile senkronlanacak
            evidence_delay_s=5.0,         # PPE ile aynı periyot (istersen 2.0 yap)
            also_console=True
        )

        # Fire için logging bayrağı (checkbox ile senkron)
        self.ENABLE_FIRE_LOGGING = self.var_fire.get()
         # ---- PPE LOGGING BAYRAĞI ----
        self.ENABLE_PPE_LOGGING = self.var_ppe.get()  # checkbox ile senkron başlasın

    # ------------------------- Yardımcılar -------------------------
    def _set_status(self, text: str):
        self.status_lbl.config(text=f"Durum: {text}")

    def _on_exit(self):
        self.stop()
        self.root.destroy()
        
    # ------------------ Model checkbox güncelle ------------------
    def update_model_flags(self):
        self.mm.set_enabled("ppe",  self.var_ppe.get())
        self.mm.set_enabled("path", self.var_path.get())
        self.mm.set_enabled("fire", self.var_fire.get())

        # PPE loglama bayrağı
        new_ppe  = bool(self.var_ppe.get())
        was_ppe  = bool(getattr(self, "ENABLE_PPE_LOGGING", True))
        self.ENABLE_PPE_LOGGING = new_ppe
        if was_ppe and not new_ppe:
            try: self.tracker.flush()
            except: pass
            if hasattr(self, "_last_res"):
                self._last_res["detections"] = [d for d in self._last_res.get("detections", [])
                    if not (str(d.cls_name).startswith("ppe:") or getattr(d, "source", "").lower() == "ppe")]
                self._last_res["masks"] = [m for m in self._last_res.get("masks", [])
                    if not (str(m.cls_name).lower().startswith("ppe:") or getattr(m, "source", "").lower() == "ppe")]

        # FIRE loglama bayrağı
        new_fire = bool(self.var_fire.get())
        was_fire = bool(getattr(self, "ENABLE_FIRE_LOGGING", True))
        self.ENABLE_FIRE_LOGGING = new_fire
        if was_fire and not new_fire:
            try: self.fire_logger.flush()
            except: pass
            if hasattr(self, "_last_res"):
                self._last_res["detections"] = [d for d in self._last_res.get("detections", [])
                    if not (str(d.cls_name).startswith("fire:") or getattr(d, "source", "").lower() == "fire")]
                self._last_res["masks"] = [m for m in self._last_res.get("masks", [])
                    if not (str(m.cls_name).lower().startswith("fire:") or getattr(m, "source", "").lower() == "fire")]


    # ------------------------- Kaynak seçim & dosya diyalogu -------------------------
    def on_source_change(self, event=None):
        sel = self.source_type.get()
        if sel in ("Resim", "Video"):
            self.btn_select.config(state="normal")
        else:
            self.btn_select.config(state="disabled")
            self.source_path = None

        try:
            mode = "photo" if sel == "Resim" else "video"
            self.tracker.source_kind = mode
            self.fire_logger.source_kind = mode
        except Exception:
            pass

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

        try:
            mode = "photo" if sel == "Resim" else "video"
            self.tracker.source_kind = mode
            self.fire_logger.source_kind = mode
        except Exception:
            pass

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

        # Tek kare cache'ini temizle (kaynaklar arasında geçişte güvenli)
        if hasattr(self, "_static_image"):
            self._static_image = None

        self._set_status("Durduruldu")

        try: self.tracker.flush()
        except: pass
        try: self.fire_logger.flush()
        except: pass

    # ------------------------- Görüntü işleme döngüsü -------------------------
    def update_frame(self):
        if not self.running:
            return

        # ── 0) Ayarlar / cache
        DETECT_EVERY = getattr(self, "DETECT_EVERY", 2)  # her N karede bir infer
        if not hasattr(self, "_last_res"):
            self._last_res = {"detections": [], "masks": []}

        # ── 1) Kaynaktan HAM kare
        sel = self.source_type.get()  # "Resim" | "Video" | "Webcam" ...
        if sel == "Resim":
            if not hasattr(self, "_static_image"):
                img = cv2.imread(self.source_path)
                if img is None:
                    self._set_status("Resim okunamadı!")
                    self.running = False
                    return
                self._static_image = img
            frame_raw = self._static_image.copy()
        else:
            if self.cap is None:
                return
            ok, frm = self.cap.read()
            if not ok:
                self.stop(); return
            frame_raw = frm

        H0, W0 = frame_raw.shape[:2]

        # ── 2) Inference: her N karede bir
        cur_idx = self.frame_idx
        do_infer = (cur_idx % max(1, DETECT_EVERY) == 0)
        if do_infer:
            res = self.mm.infer(frame_raw, frame_idx=cur_idx)
            self._last_res = res
        else:
            res = self._last_res
        self.frame_idx += 1

        # ── 3) Yardımcılar
        def _norm(name: str) -> str:
            n = str(name).split(":", 1)[-1].lower().replace("-", " ")
            return "_".join(n.split())

        def _src_and_name(name: str):
            # "ppe:helmet" -> ("ppe","helmet")  |  "fire:smoke" -> ("fire","smoke")  |  "person"->("","person")
            parts = str(name).split(":", 1)
            if len(parts) == 2:
                return parts[0].lower(), _norm(parts[1])
            return "", _norm(name)

        # Çizim kuralları (kolay genişletilir)
        POS_PPE = {"helmet", "vest"}
        NEG_PPE = {"not_helmet", "not_vest"}
        DRAW_RULES = {
            "ppe": {
                "keep": POS_PPE | NEG_PPE,
                "color": lambda nm: (0, 200, 0) if nm in POS_PPE else (0, 0, 255),
                "label": True,
            },
            "fire": {
                "keep": None,
                "color": lambda nm: (0, 0, 255) if nm == "fire" else (0, 140, 255),
                "label": True,
            },
            # "forklift": {"keep": {"forklift"}, "color": lambda nm: (255, 200, 0), "label": True},
        }

        ppe_classes = {"helmet", "vest", "not_helmet", "not_vest"}

        # Bu karede gerçekten PPE dedeksiyonu var mı?
        ppe_dets = []
        for d in res["detections"]:
            src, nm = _src_and_name(d.cls_name)
            if src == "ppe" and nm in ppe_classes:
                ppe_dets.append(d)
        
        ppe_active = bool(getattr(self, "ENABLE_PPE_LOGGING", True)) and len(ppe_dets) > 0

        # PPE kapatıldıysa, açık bir epizot varsa nezaketle kapat (tek seferlik)
        prev = getattr(self, "_ppe_active_prev", None)
        if prev is True and ppe_active is False:
            try:
                self.tracker.flush()
            except Exception:
                pass
        self._ppe_active_prev = ppe_active

        # ── 4) LOG (yalnızca PPE’ye göre missing/conf çıkar)
        if ppe_active:
            # frame içi max skorları sadece PPE kutularından topla
            max_conf_ppe = {"helmet": 0.0, "vest": 0.0, "not_helmet": 0.0, "not_vest": 0.0}
            for d in ppe_dets:
                _, nm = _src_and_name(d.cls_name)  # src zaten 'ppe'
                c = float(d.conf)
                if c > max_conf_ppe[nm]:
                    max_conf_ppe[nm] = c

            # eksikler: pozitif yoksa veya not_* varsa
            missing = []
            if (max_conf_ppe["helmet"] <= 0.0) or (max_conf_ppe["not_helmet"] > 0.0):
                missing.append("helmet")
            if (max_conf_ppe["vest"]   <= 0.0) or (max_conf_ppe["not_vest"]   > 0.0):
                missing.append("vest")

            try:
                self.tracker.process_frame(
                    frame_id=int(cur_idx),
                    frame_ts=time.time(),
                    observations=[{
                        "person_id": None,
                        "missing": missing,
                        "conf": {k: v for k, v in max_conf_ppe.items() if v > 0.0}
                    }]
                )
            except Exception:
                pass
        
        # ── 4.b) FIRE LOG
        fire_classes = {"fire", "smoke"}
        fire_dets = []
        for d in res["detections"]:
            parts = str(d.cls_name).split(":", 1)
            if len(parts) == 2:
                src, nm = parts[0].lower(), parts[1].lower().replace("-", " ").replace(" ", "_")
            else:
                src, nm = "", parts[0].lower().replace("-", " ").replace(" ", "_")
            if not src and getattr(d, "source", None):
                src = str(getattr(d, "source")).lower()
            if src == "fire" and nm in fire_classes:
                fire_dets.append(d)

        fire_active = bool(getattr(self, "ENABLE_FIRE_LOGGING", True)) and len(fire_dets) > 0
        # Kapatıldıysa, bir önce açık idiyse flush (nazik kapanış)
        prev_fire = getattr(self, "_fire_active_prev", None)
        if prev_fire is True and fire_active is False:
            try: self.fire_logger.flush()
            except: pass
        self._fire_active_prev = fire_active

        if fire_active:
            max_conf_fire = {"fire": 0.0, "smoke": 0.0}
            for d in fire_dets:
                parts = str(d.cls_name).split(":", 1)
                nm = (parts[1] if len(parts)==2 else parts[0]).lower().replace("-", " ").replace(" ", "_")
                c = float(d.conf)
                if c > max_conf_fire[nm]:
                    max_conf_fire[nm] = c
            hazards = [k for k, v in max_conf_fire.items() if v > 0.0]
            try:
                self.fire_logger.process_frame(
                    frame_id=int(cur_idx),
                    frame_ts=time.time(),
                    observations=[{
                        "hazards": hazards,
                        "conf": {k: v for k, v in max_conf_fire.items() if v > 0.0}
                    }]
                )
            except Exception:
                pass

        # ── 5) ÇİZİM: ham boyutta
        out_raw = frame_raw.copy()

        # 5.a) Mask overlay'ler
        def _overlay_mask(mask_bool, color_bgr, alpha=0.45):
            color = np.array(color_bgr, dtype=np.float32)
            out_raw[mask_bool] = (alpha * color + (1.0 - alpha) * out_raw[mask_bool]).astype(np.uint8)

        for m in res.get("masks", []):
            src, nm = _src_and_name(m.cls_name)
            mask = m.mask.astype(bool)
            mh, mw = mask.shape[:2]
            if (mh, mw) != (H0, W0):
                mask = cv2.resize(mask.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST).astype(bool)

            if src == "" and nm.startswith("path"):
                _overlay_mask(mask, (0, 180, 80))            # yeşil
            elif src == "fire" and nm == "fire":
                _overlay_mask(mask, (0, 0, 255))             # kırmızı
            elif src == "fire" and nm == "smoke":
                _overlay_mask(mask, (0, 140, 255))           # turuncu

        # 5.b) BBox çizimleri (kurallara göre)
        for d in res["detections"]:
            src, nm = _src_and_name(d.cls_name)
            rule = DRAW_RULES.get(src)
            if rule is None:
                continue
            if (rule.get("keep") is not None) and (nm not in rule["keep"]):
                continue
            x1, y1, x2, y2 = map(int, d.xyxy)
            color = rule["color"](nm)
            cv2.rectangle(out_raw, (x1, y1), (x2, y2), color, 2)
            if rule.get("label", True):
                lbl = f"{src}:{nm} {d.conf:.2f}"
                cv2.putText(out_raw, lbl, (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # ── 6) Gösterim için küçült (tek kez)
        out = self._resize_keep_aspect(out_raw, (VIEW_W, VIEW_H))

        # ── 7) FPS
        now = time.time()
        dt = now - self.last_tick
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps > 0 else (1.0 / dt)
        self.last_tick = now
        self.fps_lbl.config(text=f"FPS: {self.fps:.1f}")

        # ── 8) TK gösterim
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # ── 9) Döngü (fotoğrafta da devam: 5 sn'lik periyodik kayıtlar için gerekli)
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
