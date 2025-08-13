# app/run_cam.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from detectors.yolo_v8 import YOLOv8Detector

def main():
    det = YOLOv8Detector()
    det.load("yolov8n.pt")  # ilk deneme için hazır model (insan dahil COCO sınıfları)

    cap = cv2.VideoCapture(0)  # Bilgisayar kamerası
    if not cap.isOpened():
        raise RuntimeError("Kamera açılamadı.")

    names = det.class_names()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = det.predict(frame, conf=0.35, imgsz=640)
        out = frame.copy()

        for d in detections:
            x1, y1, x2, y2 = d.box
            cls_name = names.get(d.cls_id, str(d.cls_id))
            label = f"{cls_name} {d.score:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(out, label, (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        cv2.imshow("YOLOv8 Webcam", out)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
