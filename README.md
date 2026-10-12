# Emotion Detector — IBM Final Project

A Flask web app that detects emotions (anger, disgust, fear, joy, sadness) in
text and reports the dominant emotion, using the Watson NLP `EmotionPredict`
service.

## Layout
- `EmotionDetection/emotion_detection.py` — `emotion_detector(text_to_analyze)`
- `EmotionDetection/__init__.py` — exposes `emotion_detector` at package level
- `server.py` — Flask app (`/` page, `/emotionDetector?textToAnalyze=...`)
- `templates/index.html`, `static/mywebscript.js` — web UI
- `test_emotion_detection.py` — unit tests (HTTP layer mocked)

## Setup
```
pip install -r requirements.txt
```
The Watson endpoint defaults to the IBM Skills Network lab URL. Override it
without editing code via the `WATSON_EMOTION_URL` environment variable. No keys
or tokens are stored in this repository.

## Run
```
python3 server.py        # then open http://localhost:5000
```
Blank input, or an input the service rejects with HTTP 400, shows
`Invalid text! Please try again!`.

## Test and lint
```
python3 -m unittest test_emotion_detection -v
python3 -m pylint server.py
```
