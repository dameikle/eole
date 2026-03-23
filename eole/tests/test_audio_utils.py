import unittest

from eole.inputters.audio_utils import merge_vad_segments


class TestMergeVADSegments(unittest.TestCase):
    def test_empty_segments(self):
        self.assertEqual(merge_vad_segments([]), [])

    def test_merge_adjacent_segments(self):
        segments = [
            {"start": 0.0, "end": 1.0},
            {"start": 1.2, "end": 2.0},
        ]
        merged = merge_vad_segments(segments, max_chunk_seconds=30.0, max_merge_gap_seconds=0.5)
        self.assertEqual(merged, [{"start": 0.0, "end": 2.0}])

    def test_split_large_gap(self):
        segments = [
            {"start": 0.0, "end": 1.0},
            {"start": 4.0, "end": 5.0},
        ]
        merged = merge_vad_segments(segments, max_chunk_seconds=30.0, max_merge_gap_seconds=1.0)
        self.assertEqual(merged, segments)

    def test_split_oversized_segment(self):
        segments = [{"start": 0.0, "end": 65.0}]
        merged = merge_vad_segments(segments, max_chunk_seconds=30.0, max_merge_gap_seconds=1.0)
        self.assertEqual(
            merged,
            [
                {"start": 0.0, "end": 30.0},
                {"start": 30.0, "end": 60.0},
                {"start": 60.0, "end": 65.0},
            ],
        )


if __name__ == "__main__":
    unittest.main()
