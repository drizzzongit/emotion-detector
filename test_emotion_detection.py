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
