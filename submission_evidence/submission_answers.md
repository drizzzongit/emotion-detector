# Submission answers

> **Disclosures.** (a) Question numbering is inferred from the task order in the master prompt (Q1 README, Q6 `__init__.py`, Q11/Q14 screenshots); confirm against the live Coursera form. (b) The Watson endpoint is blocked from this sandbox, so outputs for Q3, Q5, Q7 and both screenshots were produced by running the real code against a **local stand-in service** following the Watson contract, not the real Watson service. Re-run them on the IBM lab environment if your grader expects real service scores. (c) Unit tests mock the HTTP layer.


## Question 1 — README.md public URL

**PENDING — repo not yet published.** Expected form once pushed: `https://github.com/drizzzongit/emotion-detector/blob/main/README.md` (UNVERIFIED; do not submit until it resolves).

## Question 2 — emotion_detection.py (Task 2)

```
import requests

def emotion_detector(text_to_analyse):
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    myobj = {"raw_document": {"text": text_to_analyse}}
    response = requests.post(url, json=myobj, headers=headers, timeout=30)
    return response.text
```


## Question 3 — Application creation test output

```
$ python3 -c "from emotion_detection import emotion_detector; print(emotion_detector(\"I love my life\"))"
{"emotionPredictions": [{"emotion": {"anger": 0.02, "disgust": 0.02, "fear": 0.02, "joy": 0.9, "sadness": 0.02}}]}
```


## Question 4 — Formatted emotion_detector (Task 3)

```
import json
import requests

def emotion_detector(text_to_analyze):
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    payload = {"raw_document": {"text": text_to_analyze}}
    response = requests.post(url, json=payload, headers=headers, timeout=30)

    scores = json.loads(response.text)["emotionPredictions"][0]["emotion"]
    names = ("anger", "disgust", "fear", "joy", "sadness")
    result = {name: scores[name] for name in names}
    result["dominant_emotion"] = max(names, key=result.get)
    return result
```


## Question 5 — Formatting test output

```
$ python3 -c "from emotion_detection import emotion_detector; print(emotion_detector(\"I am so sad about this\"))"
{'anger': 0.02, 'disgust': 0.02, 'fear': 0.02, 'joy': 0.02, 'sadness': 0.9, 'dominant_emotion': 'sadness'}
```


## Question 6 — EmotionDetection/__init__.py public URL

**PENDING.** Expected form: `https://github.com/drizzzongit/emotion-detector/blob/main/EmotionDetection/__init__.py` (UNVERIFIED).

## Question 7 — Packaging test output

```
$ python3
>>> from EmotionDetection.emotion_detection import emotion_detector
>>> from EmotionDetection import emotion_detector as pkg_detector
>>> pkg_detector is emotion_detector
True
{'anger': 0.02, 'disgust': 0.02, 'fear': 0.02, 'joy': 0.9, 'sadness': 0.02, 'dominant_emotion': 'joy'}
```


## Question 8 — Unit test source

```
"""Unit tests for the EmotionDetection package (HTTP layer mocked)."""
import json
import unittest
from unittest.mock import MagicMock, patch

from EmotionDetection import emotion_detector

CASES = {
    "I am glad this happened": "joy",
    "I am really mad about this": "anger",
    "I feel disgusted just hearing about this": "disgust",
    "I am so sad about this": "sadness",
    "I am really afraid that this will happen": "fear",
}

FIXTURES = {
    "joy": {"anger": 0.01, "disgust": 0.02, "fear": 0.03,
            "joy": 0.93, "sadness": 0.05},
    "anger": {"anger": 0.88, "disgust": 0.10, "fear": 0.02,
              "joy": 0.01, "sadness": 0.04},
    "disgust": {"anger": 0.12, "disgust": 0.91, "fear": 0.03,
                "joy": 0.01, "sadness": 0.06},
    "sadness": {"anger": 0.02, "disgust": 0.03, "fear": 0.04,
                "joy": 0.01, "sadness": 0.95},
    "fear": {"anger": 0.03, "disgust": 0.02, "fear": 0.90,
             "joy": 0.01, "sadness": 0.07},
}


def fake_response(status_code, body=None):
    """Build a stand-in for requests.Response."""
    response = MagicMock()
    response.status_code = status_code
    response.text = json.dumps(body) if body is not None else ""
    if status_code >= 400:
        response.raise_for_status.side_effect = RuntimeError("HTTP error")
    return response


def watson_body(scores):
    """Wrap scores in the documented Watson response structure."""
    return {"emotionPredictions": [{"emotion": scores,
                                    "target": "", "emotionMentions": []}]}


class TestEmotionDetector(unittest.TestCase):
    """Tests for emotion_detector."""

    def test_dominant_emotion_for_course_inputs(self):
        """Each course sentence maps to its expected dominant emotion."""
        for text, expected in CASES.items():
            with self.subTest(text=text):
                body = watson_body(FIXTURES[expected])
                with patch("EmotionDetection.emotion_detection.requests.post",
                           return_value=fake_response(200, body)):
                    result = emotion_detector(text)
                self.assertEqual(result["dominant_emotion"], expected)

    def test_output_keys_and_scores(self):
        """Result has exactly six keys and preserves original scores."""
        body = watson_body(FIXTURES["joy"])
        with patch("EmotionDetection.emotion_detection.requests.post",
                   return_value=fake_response(200, body)):
            result = emotion_detector("I am glad this happened")
        self.assertEqual(
            set(result),
            {"anger", "disgust", "fear", "joy", "sadness", "dominant_emotion"},
        )
        for name, score in FIXTURES["joy"].items():
            self.assertEqual(result[name], score)

    def test_request_payload(self):
        """The text is sent in the documented raw_document payload."""
        body = watson_body(FIXTURES["joy"])
        with patch("EmotionDetection.emotion_detection.requests.post",
                   return_value=fake_response(200, body)) as mock_post:
            emotion_detector("hello")
        self.assertEqual(mock_post.call_args.kwargs["json"],
                         {"raw_document": {"text": "hello"}})

    def test_status_400_returns_none(self):
        """HTTP 400 yields None."""
        with patch("EmotionDetection.emotion_detection.requests.post",
                   return_value=fake_response(400)):
            self.assertIsNone(emotion_detector(""))

    def test_server_error_is_not_swallowed(self):
        """HTTP 500 raises instead of being parsed as a prediction."""
        with patch("EmotionDetection.emotion_detection.requests.post",
                   return_value=fake_response(500)):
            with self.assertRaises(RuntimeError):
                emotion_detector("anything")

    def test_malformed_body_raises(self):
        """A 200 response without emotionPredictions raises KeyError."""
        with patch("EmotionDetection.emotion_detection.requests.post",
                   return_value=fake_response(200, {"unexpected": 1})):
            with self.assertRaises(KeyError):
                emotion_detector("anything")


if __name__ == "__main__":
    unittest.main()
```


## Question 9 — Unit test result

```
$ python3 -m unittest test_emotion_detection -v
test_dominant_emotion_for_course_inputs (test_emotion_detection.TestEmotionDetector.test_dominant_emotion_for_course_inputs)
Each course sentence maps to its expected dominant emotion. ... ok
test_malformed_body_raises (test_emotion_detection.TestEmotionDetector.test_malformed_body_raises)
A 200 response without emotionPredictions raises KeyError. ... ok
test_output_keys_and_scores (test_emotion_detection.TestEmotionDetector.test_output_keys_and_scores)
Result has exactly six keys and preserves original scores. ... ok
test_request_payload (test_emotion_detection.TestEmotionDetector.test_request_payload)
The text is sent in the documented raw_document payload. ... ok
test_server_error_is_not_swallowed (test_emotion_detection.TestEmotionDetector.test_server_error_is_not_swallowed)
HTTP 500 raises instead of being parsed as a prediction. ... ok
test_status_400_returns_none (test_emotion_detection.TestEmotionDetector.test_status_400_returns_none)
HTTP 400 yields None. ... ok

----------------------------------------------------------------------
Ran 6 tests in 0.005s

OK
```


## Question 10 — Flask server.py (Task 6)

```
"""Flask server exposing the emotion detector."""
from flask import Flask, render_template, request

from EmotionDetection import emotion_detector

app = Flask("Emotion Detector")


@app.route("/emotionDetector")
def emotion_detection_route():
    """Analyze the submitted text and return a formatted description."""
    text_to_analyze = request.args.get("textToAnalyze")
    result = emotion_detector(text_to_analyze)
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
```


## Question 11 — Deployment screenshot

Upload `submission_evidence/6b_deployment_test.png`.

## Question 12 — Error-handling emotion_detector (Task 7a)

```
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

    Returns None when the service rejects the input with HTTP 400.
    """
    payload = {"raw_document": {"text": text_to_analyze}}
    response = requests.post(
        URL, json=payload, headers=HEADERS, timeout=TIMEOUT_SECONDS
    )

    if response.status_code == 400:
        return None
    response.raise_for_status()

    scores = json.loads(response.text)["emotionPredictions"][0]["emotion"]
    result = {name: scores[name] for name in EMOTIONS}
    result["dominant_emotion"] = max(EMOTIONS, key=result.get)
    return result
```


## Question 13 — Error-handling server.py (Task 7b)

```
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
```


## Question 14 — Error-handling interface screenshot

Upload `submission_evidence/7c_error_handling_interface.png`.

## Question 15 — Modified server.py for static analysis

```
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
```


## Question 16 — Pylint output

```
$ python3 -m pylint server.py

--------------------------------------------------------------------
Your code has been rated at 10.00/10 (previous run: 10.00/10, +0.00)
```
