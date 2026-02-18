import cv2
import time
import os
import csv
import winsound
from ultralytics import YOLO

from roi_config import RESTRICTED_ZONE
from intrusion_logic import is_inside_restricted_zone

# =========================
# Load trained YOLO model
# =========================
MODEL_PATH = "runs/detect/smart_cctv_v13/weights/best.pt"
model = YOLO(MODEL_PATH)

# =========================
# Open external webcam
# Change 0 / 1 if needed
# =========================
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ Webcam not accessible")
    exit()

# =========================
# Capture & alert settings
# =========================
SAVE_DIR = "intrusion_detection/captures"
os.makedirs(SAVE_DIR, exist_ok=True)

LOG_FILE = "intrusion_detection/intrusion_log.csv"

CAPTURE_INTERVAL = 30  # seconds
last_capture_time = 0

BEEP_FREQUENCY = 1500  # Hz
BEEP_DURATION = 700    # ms

# =========================
# Create CSV log if missing
# =========================
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Time", "Event", "Image"])

print("🚀 Smart CCTV – Intrusion Detection Started")
print("📸 Image saved every 30 seconds")
print("🔊 Alarm enabled")
print("📝 Logging enabled")
print("Press 'Q' to exit")

# =========================
# Main Loop
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw restricted zone
    cv2.polylines(frame, [RESTRICTED_ZONE], True, (0, 0, 255), 2)

    # YOLO inference (person only)
    results = model(frame, conf=0.4, device=0)

    intrusion_detected = False

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) != 0:  # class 0 = person
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            inside = is_inside_restricted_zone(cx, cy, RESTRICTED_ZONE)

            if inside:
                intrusion_detected = True
                color = (0, 0, 255)
                label = "INTRUSION"
            else:
                color = (0, 255, 0)
                label = "Person"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

    # =========================
    # Alert, Capture & Log
    # =========================
    current_time = time.time()

    if intrusion_detected:
        cv2.putText(frame, "⚠ RESTRICTED AREA BREACH ⚠",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3)

        if current_time - last_capture_time >= CAPTURE_INTERVAL:
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"intrusion_{timestamp}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)

            # Save image
            cv2.imwrite(filepath, frame)
            print(f"📸 Intrusion captured: {filepath}")

            # Beep alarm
            winsound.Beep(BEEP_FREQUENCY, BEEP_DURATION)

            # Log to CSV
            with open(LOG_FILE, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.strftime("%Y-%m-%d"),
                    time.strftime("%H:%M:%S"),
                    "Restricted Area Breach",
                    filename
                ])

            last_capture_time = current_time

    cv2.imshow("Smart CCTV - Intrusion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# Cleanup
# =========================
cap.release()
cv2.destroyAllWindows()
