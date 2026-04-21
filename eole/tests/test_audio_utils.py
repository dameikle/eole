import unittest

from eole.inputters.audio_utils import merge_vad_segments


class TestMergeVADSegments(unittest.TestCase):
    def test_empty_segments(self):
        self.assertEqual(merge_vad_segments([]), [])

    def test_merge_within_chunk_size(self):
        segments = [
            {"start": 0.0, "end": 1.0},
            {"start": 11.0, "end": 12.0},
        ]
        merged = merge_vad_segments(segments, max_chunk_seconds=30.0)
        self.assertEqual(merged, [{"start": 0.0, "end": 12.0}])

    def test_split_on_span_exceeding_chunk_size(self):
        segments = [
            {"start": 0.0, "end": 5.0},
            {"start": 10.0, "end": 15.0},
            {"start": 28.0, "end": 33.0},
        ]
        merged = merge_vad_segments(segments, max_chunk_seconds=30.0)
        self.assertEqual(
            merged,
            [
                {"start": 0.0, "end": 15.0},
                {"start": 28.0, "end": 33.0},
            ],
        )

    def test_single_oversized_segment_not_split(self):
        segments = [{"start": 0.0, "end": 65.0}]
        merged = merge_vad_segments(segments, max_chunk_seconds=30.0)
        self.assertEqual(merged, [{"start": 0.0, "end": 65.0}])

    def test_all_invalid_segments(self):
        segments = [
            {"start": 5.0, "end": 3.0},
            {"start": 10.0, "end": 10.0},
        ]
        self.assertEqual(merge_vad_segments(segments), [])

    def test_single_segment(self):
        segments = [{"start": 2.0, "end": 5.0}]
        merged = merge_vad_segments(segments, max_chunk_seconds=30.0)
        self.assertEqual(merged, [{"start": 2.0, "end": 5.0}])

    def test_non_zero_offset(self):
        segments = [
            {"start": 100.0, "end": 110.0},
            {"start": 115.0, "end": 120.0},
        ]
        merged = merge_vad_segments(segments, max_chunk_seconds=30.0)
        self.assertEqual(merged, [{"start": 100.0, "end": 120.0}])

    def test_overlapping_segments(self):
        segments = [
            {"start": 0.0, "end": 20.0},
            {"start": 5.0, "end": 10.0},
        ]
        merged = merge_vad_segments(segments, max_chunk_seconds=30.0)
        self.assertEqual(merged, [{"start": 0.0, "end": 20.0}])

    def test_span_exactly_equals_chunk_size(self):
        segments = [
            {"start": 0.0, "end": 10.0},
            {"start": 20.0, "end": 30.0},
        ]
        merged = merge_vad_segments(segments, max_chunk_seconds=30.0)
        self.assertEqual(merged, [{"start": 0.0, "end": 30.0}])

    def test_multiple_splits(self):
        segments = [
            {"start": 0.0, "end": 2.0},
            {"start": 5.0, "end": 7.0},
            {"start": 25.0, "end": 27.0},
            {"start": 29.0, "end": 35.0},
            {"start": 40.0, "end": 42.0},
            {"start": 60.0, "end": 68.0},
        ]
        merged = merge_vad_segments(segments, max_chunk_seconds=30.0)
        self.assertEqual(
            merged,
            [
                {"start": 0.0, "end": 27.0},
                {"start": 29.0, "end": 42.0},
                {"start": 60.0, "end": 68.0},
            ],
        )


if __name__ == "__main__":
    unittest.main()
