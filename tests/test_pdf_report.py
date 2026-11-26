import sqlite3
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from app.main import video_report


def report_database(status="done"):
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript("""
        CREATE TABLE videos(
            id INTEGER PRIMARY KEY, name TEXT, path TEXT, status TEXT, stage TEXT,
            progress REAL, fps REAL, frames INTEGER, duration REAL, width INTEGER,
            height INTEGER, analysis_stride INTEGER, error TEXT, created TEXT
        );
        CREATE TABLE detections(
            video_id INTEGER, frame INTEGER, t REAL, track_id INTEGER, cls TEXT,
            conf REAL, x1 REAL, y1 REAL, x2 REAL, y2 REAL
        );
        CREATE TABLE anomaly_reviews(
            video_id INTEGER, anomaly_id TEXT, status TEXT, note TEXT
        );
    """)
    connection.execute(
        "INSERT INTO videos VALUES(7, 'sample.mp4', 'sample.mp4', ?, 'complete', "
        "1.0, 30.0, 300, 10.0, 640, 360, 2, NULL, '2026-01-01')",
        (status,),
    )
    connection.executemany(
        "INSERT INTO detections VALUES(?,?,?,?,?,?,?,?,?,?)",
        [
            (7, 0, 0.0, 1, "car", 0.9, 10, 10, 30, 30),
            (7, 1, 1.0, 1, "car", 0.9, 12, 10, 32, 30),
            (7, 0, 0.0, 2, "person", 0.8, 50, 20, 60, 50),
        ],
    )
    return connection


class PdfReportTests(unittest.TestCase):
    def test_processed_video_returns_downloadable_pdf(self):
        connection = report_database()
        with patch("app.main.db.conn", return_value=connection):
            response = video_report(7)

        self.assertEqual(response.media_type, "application/pdf")
        self.assertEqual(
            response.headers["content-disposition"],
            'attachment; filename="nemesis-x-video-7-report.pdf"',
        )
        self.assertTrue(response.body.startswith(b"%PDF-"))
        self.assertTrue(response.body.rstrip().endswith(b"%%EOF"))
        self.assertGreater(len(response.body), 1000)
        connection.close()

    def test_unprocessed_video_cannot_generate_report(self):
        connection = report_database(status="uploaded")
        with patch("app.main.db.conn", return_value=connection):
            with self.assertRaises(HTTPException) as error:
                video_report(7)

        self.assertEqual(error.exception.status_code, 409)
        connection.close()
