import cv2

def is_inside_restricted_zone(cx, cy, zone_polygon):
    return cv2.pointPolygonTest(zone_polygon, (cx, cy), False) >= 0
