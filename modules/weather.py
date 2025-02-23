import os
import requests
from datetime import datetime
from hub_models import gemini_chat
from modules.audio import text_to_audio_file, play_audio_file
from dotenv import load_dotenv
load_dotenv()

def fetch_weather_data():
    api_key = os.environ["WEATHER_API_KEY"]
    location = "Atlanta,GA"
    today = datetime.now().strftime("%Y-%m-%d")
    base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    url = f"{base_url}/{location}/{today}/{today}"
    try:
        response = requests.get(url, params={"unitGroup": "metric", "key": api_key})
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None
def generate_weather_message():
    weather_data = fetch_weather_data()
    if weather_data is None:
        return "Unable to fetch weather data at this time."
    day = weather_data["days"][0]
    city = weather_data.get("resolvedAddress", "your area")
    prompt = (f"As your trusted weather advisor for a blind person, please summarize today's weather in {city} – "
              f"{day['description']} with a high of {day['tempmax']}°C, a low of {day['tempmin']}°C, "
              f"and an average of {day['temp']}°C – in one or two clear sentences, advise me on the most important precautions to take today.")
    try:
        response = gemini_chat.send_message(prompt)
        result = response.text.strip()
        if result.startswith(prompt):
            result = result[len(prompt):].strip()
        return result
    except Exception:
        return prompt
def weather_audio_worker():
    message = generate_weather_message()
    filename = text_to_audio_file(message)
    play_audio_file(filename)
