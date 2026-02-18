import numpy as np

# Adjust based on camera resolution (example: 640x480)
ZONES = {
    "Zone A": np.array([
        (0, 0),
        (640, 0),
        (640, 240),
        (0, 240)
    ], np.int32),

    "Zone B": np.array([
        (0, 240),
        (640, 240),
        (640, 480),
        (0, 480)
    ], np.int32)
}
