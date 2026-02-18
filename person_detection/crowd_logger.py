import csv
import os
import time

LOG_FILE = "person_detection/crowd_log.csv"

# Create CSV file with header if it does not exist
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Time", "People_Count"])


def log_crowd_count(count):
    """
    Logs crowd count with timestamp
    """
    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            time.strftime("%Y-%m-%d"),
            time.strftime("%H:%M:%S"),
            count
        ])
