import sqlite3
import unittest
from unittest.mock import patch

from app.main import (
    AnomalyReview,
    anomalies,
    model_motion_anomalies,
    motion_anomalies,
    review_anomaly,
)


def anomaly_database():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript("""
        CREATE TABLE videos(id INTEGER PRIMARY KEY, status TEXT);
        INSERT INTO videos VALUES(7, 'done');
        CREATE TABLE detections(
            video_id INTEGER, frame INTEGER, t REAL, track_id INTEGER, cls TEXT,
            conf REAL, x1 REAL, y1 REAL, x2 REAL, y2 REAL
        );
        INSERT INTO detections VALUES
            (7, 0, 0.0, 1, 'car', 0.9, 10, 10, 30, 30),
            (7, 1, 0.2, 1, 'car', 0.9, 12, 10, 32, 30);
        CREATE TABLE anomaly_reviews(
            video_id INTEGER NOT NULL, anomaly_id TEXT NOT NULL,
            status TEXT NOT NULL, note TEXT, updated TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(video_id, anomaly_id)
        );
    """)
    return connection


class AnomalyReviewTests(unittest.TestCase):
    def test_empty_candidates_include_threshold_and_review_state(self):
        connection = anomaly_database()
        with patch("app.main.db.conn", return_value=connection):
            result = anomalies(7)

        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["model_candidates"], [])
        self.assertEqual(result["eligibility"]["statistical_minimum_tracks_per_class"], 5)
        self.assertEqual(result["eligibility"]["model_minimum_tracks_per_class"], 5)
        self.assertEqual(result["review_summary"], {
            "total": 0, "pending": 0, "confirmed": 0, "dismissed": 0,
        })
        self.assertEqual(result["eligibility"]["classes"][0]["qualifying_tracks"], 0)
        connection.close()

    def test_small_eligible_cohort_produces_reviewable_candidates(self):
        connection = anomaly_database()
        for track_id, speed in enumerate((1, 2, 2.5, 3, 10), start=10):
            connection.executemany(
                "INSERT INTO detections VALUES(?,?,?,?,?,?,?,?,?,?)",
                [
                    (7, frame, float(frame), track_id, "person", 0.9,
                     10 + speed * frame, 10, 30 + speed * frame, 30)
                    for frame in range(8)
                ],
            )

        statistical = motion_anomalies(connection, 7)
        model = model_motion_anomalies(connection, 7)

        self.assertTrue(any(item["track_id"] == 14 for item in statistical))
        self.assertTrue(any(item["track_id"] == 14 for item in model))
        self.assertTrue(all(item["class_track_count"] == 5 for item in statistical + model))
        connection.close()

    def test_review_action_persists_candidate_status_and_note(self):
        connection = anomaly_database()
        candidate = {"id": "speed-car-1"}
        with patch("app.main.db.conn", return_value=connection), \
             patch("app.main.motion_anomalies", return_value=[candidate]), \
             patch("app.main.model_motion_anomalies", return_value=[]):
            result = review_anomaly(
                7, candidate["id"], AnomalyReview(status="confirmed", note="Checked timestamp"),
            )

        self.assertEqual(result["status"], "confirmed")
        row = connection.execute(
            "SELECT status,note FROM anomaly_reviews WHERE video_id=7 AND anomaly_id=?",
            (candidate["id"],),
        ).fetchone()
        self.assertEqual(tuple(row), ("confirmed", "Checked timestamp"))
        connection.close()
