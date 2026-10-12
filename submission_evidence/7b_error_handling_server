"""Flask server exposing the emotion detector."""
from flask import Flask, render_template, request

from EmotionDetection import emotion_detector

app = Flask("Emotion Detector")

INVALID_TEXT_MESSAGE = "Invalid text! Please try again!"


@app.route("/emotionDetector")
def emotion_detection_route():
    """Analyze the submitted text and return a formatted description."""
    text_to_analyze = request.args.get("textToAnalyze", "")
    if not text_to_analyze.strip():
        return INVALID_TEXT_MESSAGE

    result = emotion_detector(text_to_analyze)
    if result is None or result["dominant_emotion"] is None:
        return INVALID_TEXT_MESSAGE

    return (
        "For the given statement, the system response is "
        f"'anger': {result['anger']}, 'disgust': {result['disgust']}, "
        f"'fear': {result['fear']}, 'joy': {result['joy']} and "
        f"'sadness': {result['sadness']}. "
        f"The dominant emotion is {result['dominant_emotion']}."
    )


@app.route("/")
def render_index_page():
    """Render the main page."""
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
