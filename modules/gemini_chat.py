from hub_models import gemini_chat
def generate_interaction_summary(emotion_stats):
    if not emotion_stats:
        return "No interactions were recorded today."
    summary_text = ", ".join([f"{emo}: {cnt}" for emo, cnt in emotion_stats.items()])
    prompt = (f"Based on today's interactions, the detected emotions were: {summary_text}. "
              "Provide a concise one or two sentence summary of the overall emotional tone.")
    try:
        response = gemini_chat.send_message(prompt)
        result = response.text.strip()
        if result.startswith(prompt):
            result = result[len(prompt):].strip()
        return result
    except Exception:
        return "Unable to generate an interaction summary at this time."
