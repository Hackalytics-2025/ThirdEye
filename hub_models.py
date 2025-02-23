import os
import torch
import cv2
from transformers import pipeline
import google.generativeai as genai
import torch.nn.functional as F
from dotenv import load_dotenv
load_dotenv()

yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo.to('cuda')
caption_pipeline = pipeline("image-to-text", model="salesforce/blip-image-captioning-base", device=0)
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
midas.to('cuda')
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
midas_transform = midas_transforms.default_transform
emotion_detector = pipeline("image-classification", model="dima806/facial_emotions_image_detection", device=0)
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
GEMINI_CONFIG = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash-8b", generation_config=GEMINI_CONFIG)
gemini_chat = gemini_model.start_chat(history=[])
