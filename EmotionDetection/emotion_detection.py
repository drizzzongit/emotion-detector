"""Emotion detection using the Watson NLP EmotionPredict service."""
import json
import os

import requests

URL = os.environ.get(
    "WATSON_EMOTION_URL",
    "https://sn-watson-emotion.labs.skills.network/v1/"
    "watson.runtime.nlp.v1/NlpService/EmotionPredict",
)
HEADERS = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
}
EMOTIONS = ("anger", "disgust", "fear", "joy", "sadness")
TIMEOUT_SECONDS = 30


def emotion_detector(text_to_analyze):
    """Return the five emotion scores and the dominant emotion for a text.

    When the service rejects the input with HTTP 400, every score and the
    dominant emotion are returned as None.
    """
    payload = {"raw_document": {"text": text_to_analyze}}
    response = requests.post(
        URL, json=payload, headers=HEADERS, timeout=TIMEOUT_SECONDS
    )

    if response.status_code == 400:
        result = dict.fromkeys(EMOTIONS)
        result["dominant_emotion"] = None
        return result
    response.raise_for_status()

    scores = json.loads(response.text)["emotionPredictions"][0]["emotion"]
    result = {name: scores[name] for name in EMOTIONS}
    result["dominant_emotion"] = max(EMOTIONS, key=result.get)
    return result
