import time
import os
from gtts import gTTS
import pygame
from modules.utils import remove_file_with_retry
pygame.mixer.pre_init(44100, -16, 2, 512)
pygame.init()
def text_to_audio_file(text, language="en", accent="com", filename=None):
    if filename is None:
        filename = f"speech_{int(time.time()*1000)}.mp3"
    tts = gTTS(text=text, lang=language, slow=False, tld=accent)
    tts.save(filename)
    return filename
def play_audio_file(audio_filename):
    try:
        pygame.mixer.music.load(audio_filename)
    except Exception:
        return
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.music.stop()
    pygame.mixer.music.unload()
    time.sleep(2.0)
    remove_file_with_retry(audio_filename)
def speak_text(text):
    filename = text_to_audio_file(text)
    play_audio_file(filename)
def alert_audio(frame):
    alert_msg = "Attention! There is something right in front of you that is very close!"
    filename = text_to_audio_file(alert_msg)
    play_audio_file(filename)
def process_and_play_audio(text):
    filename = text_to_audio_file(text)
    play_audio_file(filename)
