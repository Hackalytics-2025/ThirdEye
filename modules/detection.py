import numpy as np
import cv2
import torch
from hub_models import yolo, emotion_detector
from modules.utils import get_grid_location
def get_yolo_detections(image, threshold=0.6):
    results = yolo(image)
    df = results.pandas().xyxy[0]
    height, width, _ = image.shape
    total_area = width * height
    center = (width / 2, height / 2)
    detections = []
    for _, row in df.iterrows():
        if row["confidence"] < threshold:
            continue
        xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
        if ((xmax - xmin) * (ymax - ymin)) / total_area < 0.15:
            continue
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        if np.sqrt((cx - center[0])**2 + (cy - center[1])**2) / np.sqrt((width/2)**2 + (height/2)**2) > 0.25:
            continue
        detections.append(f"{row['name']} at {get_grid_location(cx, cy, width, height)}")
    return detections
def draw_detections(frame, df, threshold=0.6):
    for _, row in df.iterrows():
        if row["confidence"] < threshold:
            continue
        x1, y1, x2, y2 = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{row['name']} {row['confidence']:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame
def detect_emotion(frame):
    results = yolo(frame)
    df = results.pandas().xyxy[0]
    persons = df[df['name'] == 'person']
    if persons.empty:
        return None
    row = persons.iloc[0]
    x1, y1, x2, y2 = map(int, (row['xmin'], row['ymin'], row['xmax'], row['ymax']))
    face_crop = frame[y1:int(y1 + 0.3*(y2-y1)), x1:x2]
    if face_crop.size == 0:
        return None
    from PIL import Image
    pil_img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
    result = emotion_detector(pil_img)
    if result:
        return result[0]['label']
    return None
