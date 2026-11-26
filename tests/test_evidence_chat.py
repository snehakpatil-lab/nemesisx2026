import sqlite3
import unittest
from unittest.mock import patch

from app.main import Q, chat


def run_chat(question):
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript("""
        CREATE TABLE videos(id INTEGER PRIMARY KEY, status TEXT, duration REAL);
        CREATE TABLE detections(
            video_id INTEGER, frame INTEGER, t REAL, track_id INTEGER, cls TEXT,
            conf REAL, x1 REAL, y1 REAL, x2 REAL, y2 REAL
        );
        INSERT INTO videos VALUES(7, 'done', 10.0);
        INSERT INTO detections VALUES
            (7, 0, 0.0, 1, 'car', 0.9, 10, 10, 30, 30),
            (7, 1, 1.0, 1, 'car', 0.9, 12, 10, 32, 30),
            (7, 0, 0.0, 2, 'person', 0.9, 50, 20, 60, 50);
    """)
    with patch("app.main.db.conn", return_value=connection):
        return chat(Q(question=question, video_id=7))


class EvidenceChatTests(unittest.TestCase):
    def test_video_summary_uses_tracks_and_requested_word_limit(self):
        result = run_chat("What happened in this video? Explain in 100 words")

        self.assertIn("road-traffic scene", result["answer"])
        self.assertIn("1 car", result["answer"])
        self.assertNotIn("INSUFFICIENT EVIDENCE", result["answer"])
        self.assertLessEqual(len(result["answer"].split()), 100)
        self.assertTrue(result["evidence"])

    def test_accident_question_explains_limits_instead_of_generic_refusal(self):
        result = run_chat("Explain the acciendent situation")

        self.assertIn("accident cannot be verified", result["answer"])
        self.assertIn("collision detections", result["answer"])
        self.assertNotIn("INSUFFICIENT EVIDENCE", result["answer"])

    def test_missing_track_still_returns_insufficient_evidence(self):
        result = run_chat("Tell me about track #99")

        self.assertIn("track 99 was not observed", result["answer"])
