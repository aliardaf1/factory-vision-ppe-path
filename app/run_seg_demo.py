# app/run_seg_demo.py
import cv2, numpy as np
from detectors.yolo_v8_seg import YOLOv8Seg

def main():
    seg = YOLOv8Seg()
    seg.load("models/yellow_path_seg.pt")  # veri gelince: "models/yellow_path_seg.pt"
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): raise RuntimeError("Kamera açılamadı")

    while True:
        ok, frame = cap.read()
        if not ok: break
        mask = seg.predict_mask(frame, conf=0.25, imgsz=640)
        # görselleştir
        color = frame.copy()
        color[mask.astype(bool)] = (0, 255, 255)  # sarı overlay
        out = cv2.addWeighted(frame, 0.6, color, 0.4, 0)
        cv2.imshow("Path Seg Demo", out)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
