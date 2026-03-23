import unittest

from eole.predict.word_alignment import (
    _cap_outlier_durations,
    _interpolate_missing_word_times,
    _stabilize_word_times,
)


class TestWordAlignmentPostprocessing(unittest.TestCase):
    def test_interpolate_missing_word_times(self):
        words = [
            {"text": "hello", "start": 0.0, "end": 0.5},
            {"text": "9", "start": None, "end": None},
            {"text": "world", "start": 1.5, "end": 2.0},
        ]
        out = _interpolate_missing_word_times(words, 0.0, 2.0)
        self.assertIsNotNone(out[1]["start"])
        self.assertIsNotNone(out[1]["end"])
        self.assertGreaterEqual(out[1]["start"], 0.5)
        self.assertLessEqual(out[1]["end"], 1.5)

    def test_cap_outlier_duration(self):
        words = [{"text": "9", "start": 9.54, "end": 16.64}]
        out = _cap_outlier_durations(words, 9.0, 17.0, max_word_dur=1.5)
        self.assertLessEqual(out[0]["end"] - out[0]["start"], 1.5 + 1e-6)

    def test_stabilize_monotonic(self):
        words = [
            {"text": "a", "start": 0.3, "end": 0.2},
            {"text": "b", "start": 0.1, "end": 0.15},
            {"text": "c", "start": 0.14, "end": 0.14},
        ]
        out = _stabilize_word_times(words, 0.0, 1.0, min_word_dur=0.02)
        self.assertGreaterEqual(out[0]["end"], out[0]["start"])
        self.assertGreaterEqual(out[1]["start"], out[0]["end"])
        self.assertGreaterEqual(out[2]["start"], out[1]["end"])
        self.assertLessEqual(out[-1]["end"], 1.0)


if __name__ == "__main__":
    unittest.main()
