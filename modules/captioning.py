import cv2
import time
import numpy as np
import torch
import torch.nn.functional as F
from hub_models import caption_pipeline, midas, midas_transform, yolo
from modules.detection import get_yolo_detections, detect_emotion
def image_to_text(image_path):
    return caption_pipeline(image_path)[0]["generated_text"]
def get_depth_values(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = midas_transform(rgb).to('cuda')
    if tensor.ndim == 5:
        tensor = tensor.squeeze(0)
    with torch.no_grad():
        pred = midas(tensor)
    pred = F.interpolate(pred.unsqueeze(1), size=rgb.shape[:2], mode="bicubic", align_corners=False).squeeze()
    depth = pred.cpu().numpy()
    overall = depth.mean()
    h, w = depth.shape
    median = float(np.median(depth[h//3:2*h//3, w//3:2*w//3]))
    return median, overall
def generate_explore_text(frame, depth_margin=0.01):
    temp_image = "temp_frame.jpg"
    cv2.imwrite(temp_image, frame)
    caption = image_to_text(temp_image)
    detections = get_yolo_detections(frame)
    median, overall = get_depth_values(frame)
    if median > overall + depth_margin:
        return "Attention! There is something right in front of you that is very close!"
    if detections:
        return f"{caption}. Also detected: {', '.join(detections)}."
    return caption
def generate_conversational_text(frame, conversation_emotions):
    detected = detect_emotion(frame)
    if not detected:
        results = yolo(frame)
        df = results.pandas().xyxy[0]
        return "I don't see anyone." if df.empty else "I see a person."
    conversation_emotions.append((time.time(), detected))
    return f"The person appears to be {detected}."
def generate_audio_text(frame, mode="explore", depth_margin=0.01, conversation_emotions=None):
    if mode.lower() == "conversational":
        if conversation_emotions is None:
            conversation_emotions = []
        return generate_conversational_text(frame, conversation_emotions)
    return generate_explore_text(frame, depth_margin)
