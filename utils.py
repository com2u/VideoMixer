import os
import cv2

def get_available_cameras():
    cameras = []
    # Linux specific camera detection
    if os.path.exists('/sys/class/video4linux'):
        for d in sorted(os.listdir('/sys/class/video4linux')):
            if d.startswith('video'):
                try:
                    with open(f'/sys/class/video4linux/{d}/name', 'r') as f:
                        name = f.read().strip()
                    idx = int(d[5:])
                    cameras.append((idx, f"{name} (/{d})"))
                except: pass
    else:
        # Fallback for other OS or if sysfs is not available
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cameras.append((i, f"Camera {i}"))
                cap.release()
    return cameras