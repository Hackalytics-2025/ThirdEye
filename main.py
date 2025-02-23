import warnings
warnings.simplefilter("ignore", category=FutureWarning)
import cv2
import time
import threading
import random
import os
import numpy as np
from config import INPUT_MODE, CAMERA_INDEX, VIDEO_FILE, DEPTH_MARGIN
from modules.audio import speak_text, alert_audio, process_and_play_audio
from modules.detection import draw_detections
from modules.captioning import generate_audio_text, get_depth_values
from modules.depth import get_depth_heatmap
from modules.weather import weather_audio_worker
from modules.analysis import analyze_emotions
from modules.gemini_chat import generate_interaction_summary
from hub_models import yolo
from datetime import datetime
stop_threads = False
latest_frame = None
frame_lock = threading.Lock()
alert_event = threading.Event()
last_alert_time = 0
current_mode = "explore"
conversation_emotions = []
weather_in_progress = False
show_heatmap = False
def captioning_loop():
    global latest_frame, weather_in_progress
    while not stop_threads:
        time.sleep(random.uniform(5, 10))
        if weather_in_progress or alert_event.is_set():
            continue
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is None:
            continue
        text = generate_audio_text(frame, current_mode, DEPTH_MARGIN, conversation_emotions)
        process_and_play_audio(text)
def depth_alert_loop():
    global latest_frame, last_alert_time, weather_in_progress
    while not stop_threads:
        time.sleep(0.2)
        if weather_in_progress:
            continue
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is None:
            continue
        if current_mode.lower() == "explore":
            median, overall = get_depth_values(frame)
            if median > overall + DEPTH_MARGIN and (time.time() - last_alert_time) > 3:
                alert_event.set()
                alert_audio(frame)
                last_alert_time = time.time()
                alert_event.clear()
def main():
    global latest_frame, current_mode, stop_threads, show_heatmap, weather_in_progress, conversation_emotions
    from config import INPUT_MODE, CAMERA_INDEX, VIDEO_FILE
    if INPUT_MODE == "webcam":
        cap = cv2.VideoCapture(CAMERA_INDEX)
    elif INPUT_MODE == "video":
        cap = cv2.VideoCapture(VIDEO_FILE)
    else:
        return
    if not cap.isOpened():
        return
    cv2.namedWindow("Video Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Feed", 640, 480)
    current_mode = "explore"
    caption_thread = threading.Thread(target=captioning_loop)
    depth_thread = threading.Thread(target=depth_alert_loop)
    caption_thread.start()
    depth_thread.start()
    while True:
        ret, frame = cap.read()
        if not ret:
            if INPUT_MODE == "video":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break
        with frame_lock:
            latest_frame = frame.copy()
        results = yolo(frame)
        df = results.pandas().xyxy[0]
        frame_display = draw_detections(frame.copy(), df)
        if show_heatmap:
            heatmap = get_depth_heatmap(frame)
            h, w, _ = heatmap.shape
            overlay = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)
            cv2.line(overlay, (0, h // 3), (w, h // 3), (255, 255, 255), 1)
            cv2.line(overlay, (0, 2 * h // 3), (w, 2 * h // 3), (255, 255, 255), 1)
            cv2.line(overlay, (w // 3, 0), (w // 3, h), (255, 255, 255), 1)
            cv2.line(overlay, (2 * w // 3, 0), (2 * w // 3, h), (255, 255, 255), 1)
            cv2.imshow("Video Feed", overlay)
        else:
            cv2.imshow("Video Feed", frame_display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('e'):
            current_mode = "explore"
            print("Switched to Explore Mode.")
        elif key == ord('c'):
            current_mode = "conversational"
            print("Switched to Conversational Mode.")
        elif key == ord('w'):
            print("Fetching weather update...")
            threading.Thread(target=weather_audio_worker, daemon=True).start()
        elif key == ord('h'):
            show_heatmap = not show_heatmap
            if show_heatmap:
                print("Heatmap enabled.")
            else:
                print("Heatmap disabled.")

    stop_threads = True
    caption_thread.join()
    depth_thread.join()
    cap.release()
    cv2.destroyAllWindows()
    stats = analyze_emotions(conversation_emotions)
    summary = generate_interaction_summary(stats)
    speak_text(summary)
if __name__ == '__main__':
    main()
