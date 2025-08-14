import sys, os
import cv2
import numpy as np

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from detectors.yolo_v8 import YOLOv8Detector
from core.region_timer import RegionStayTimer
from shapely.geometry import Point, Polygon
import json
import supervision as sv 

# Sarı yol poligonunu dosyadan yükle
with open("app/path_polygon.json", "r") as f:
    path_coords = json.load(f)
path_poly = Polygon(path_coords)

timer = RegionStayTimer()
VIOL_THRESH = 5.0  # saniye

def main():
    det = YOLOv8Detector()
    det.load("yolov8n.pt")  # İlk deneme için COCO önceden eğitilmiş model

    cap = cv2.VideoCapture(0)  # Bilgisayar kamerası
    if not cap.isOpened():
        raise RuntimeError("Kamera açılamadı.")

    names = det.class_names()
    tracker = sv.ByteTrack()  # Tracking

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # YOLO tespitleri
        detections_raw = det.predict(frame, conf=0.35, imgsz=640)

        # Supervision Detections formatına çevir
        boxes = []
        scores = []
        class_ids = []
        for d in detections_raw:
            boxes.append([d.box[0], d.box[1], d.box[2], d.box[3]])  # [x1, y1, x2, y2]
            scores.append(d.score)
            class_ids.append(d.cls_id)

        detections = sv.Detections(
            xyxy=np.array(boxes),
            confidence=np.array(scores),
            class_id=np.array(class_ids)
        )

        # Sadece person (COCO class 0) filtrele
        mask = detections.class_id == 0
        detections = detections[mask]

        # Tracking
        tracks = tracker.update_with_detections(detections)

        out = frame.copy()
        for xyxy, track_id, _, _ in tracks:
            x1, y1, x2, y2 = map(int, xyxy)
            foot_x = int((x1 + x2) / 2)
            foot_y = int(y2)

            # Yol içinde mi?
            in_path = path_poly.contains(Point(foot_x, foot_y))
            sec_out = timer.update(track_id, condition=(not in_path))
            violation = sec_out >= VIOL_THRESH

            if violation:
                color = (0, 0, 255)  # Kırmızı
                label = f"ID {track_id} OUT {sec_out:.1f}s"
            else:
                color = (0, 255, 0)  # Yeşil
                label = f"ID {track_id}"

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, label, (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("YOLOv8 - Path Violation", out)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
