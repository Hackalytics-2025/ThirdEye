from collections import Counter
def analyze_emotions(conversation_emotions):
    if not conversation_emotions:
        return {}
    emotions = [emotion for _, emotion in conversation_emotions]
    stats = dict(Counter(emotions))
    return stats
