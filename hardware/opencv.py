import cv2
import numpy as np
import requests
import time

URL = "http://10.205.130.245/cam-lo.jpg"

# Bypass proxy (important for local IP)
session = requests.Session()
session.trust_env = False

def get_frame():
    r = session.get(URL, timeout=3)
    arr = np.frombuffer(r.content, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=30, detectShadows=False)

while True:
    try:
        frame = get_frame()
        if frame is None:
            continue

        frame = cv2.resize(frame, (640, 480))

        mask = bg.apply(frame)

        # Clean mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_area = 0
        best_contour = None

        for c in contours:
            area = cv2.contourArea(c)
            if area > largest_area and area > 1500:
                largest_area = area
                best_contour = c

        # Draw only ONE green box
        if best_contour is not None:
            x, y, w, h = cv2.boundingRect(best_contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3)

        cv2.imshow("ESP32 Object Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.03)

    except Exception as e:
        print("Error:", e)
        time.sleep(0.5)

cv2.destroyAllWindows()
