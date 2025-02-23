import os
from datetime import datetime

INPUT_MODE = "webcam"
CAMERA_INDEX = 1
VIDEO_FILE = "videoplayback.mp4"
CAPTION_MODEL_NAME = "salesforce/blip-image-captioning-base"
LANGUAGE = "en"
ACCENT = "com"
YOLO_CONF_THRESHOLD = 0.6
DEPTH_MARGIN = 0.01
GEMINI_CONFIG = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
LOCATION = "Atlanta,GA"
