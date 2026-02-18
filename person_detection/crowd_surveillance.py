import cv2
import numpy as np
import time
import os
import winsound
from ultralytics import YOLO

from crowd_zones import ZONES
from crowd_logger import log_crowd_count

# =========================
# Load YOLO model
# =========================
MODEL_PATH = "runs/detect/smart_cctv_v13/weights/best.pt"
model = YOLO(MODEL_PATH)

# =========================
# Webcam
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam not accessible")
    exit()

# =========================
# Crowd configuration
# =========================
MAX_CAPACITY = 20
BEEP_FREQ = 1200
BEEP_DURATION = 600

# =========================
# Heatmap
# =========================
heatmap = None
HEATMAP_ALPHA = 0.6

# =========================
# Snapshot settings
# =========================
SNAPSHOT_DIR = "person_detection/crowd_snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

SNAPSHOT_INTERVAL = 30
last_snapshot_time = 0

# =========================
# Logging settings
# =========================
LOG_INTERVAL = 60
last_log_time = 0

print("🚀 Crowd Surveillance Started")
print("Press 'Q' to exit")

# =========================
# Main Loop
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if heatmap is None:
        heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

    results = model(frame, conf=0.4, device=0)

    person_count = 0
    centers = []

    # -------------------------
    # Detection
    # -------------------------
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) != 0:
                continue

            person_count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            centers.append((cx, cy))

            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)

    # -------------------------
    # Heatmap update
    # -------------------------
    for (cx, cy) in centers:
        if 0 <= cx < heatmap.shape[1] and 0 <= cy < heatmap.shape[0]:
            heatmap[cy, cx] += 1

    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(
        heatmap_norm.astype(np.uint8),
        cv2.COLORMAP_JET
    )

    frame = cv2.addWeighted(frame, 1 - HEATMAP_ALPHA,
                            heatmap_color, HEATMAP_ALPHA, 0)

    # -------------------------
    # Zone-wise density
    # -------------------------
    zone_counts = {z: 0 for z in ZONES}

    for (cx, cy) in centers:
        for zone_name, zone_poly in ZONES.items():
            if cv2.pointPolygonTest(zone_poly, (cx, cy), False) >= 0:
                zone_counts[zone_name] += 1

    for zone_poly in ZONES.values():
        cv2.polylines(frame, [zone_poly], True, (255, 255, 255), 2)

    y = 80
    for z, c in zone_counts.items():
        cv2.putText(frame, f"{z}: {c}",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
        y += 30

    # -------------------------
    # Alert + Snapshot
    # -------------------------
    current_time = time.time()

    if person_count > MAX_CAPACITY:
        cv2.putText(frame, "⚠ CROWD OVERLOAD ⚠",
                    (160, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 4)

        winsound.Beep(BEEP_FREQ, BEEP_DURATION)

        if current_time - last_snapshot_time >= SNAPSHOT_INTERVAL:
            ts = time.strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"crowd_{ts}.jpg"
            path = os.path.join(SNAPSHOT_DIR, filename)
            cv2.imwrite(path, frame)
            print(f"📸 Snapshot saved: {path}")
            last_snapshot_time = current_time

    # -------------------------
    # Logging every minute
    # -------------------------
    if current_time - last_log_time >= LOG_INTERVAL:
        log_crowd_count(person_count)
        last_log_time = current_time

    # -------------------------
    # Display count
    # -------------------------
    cv2.putText(frame, f"People Count: {person_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 3)

    cv2.imshow("Smart CCTV - Crowd Surveillance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
