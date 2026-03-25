import unittest

import torch

from eole.predict.audio_predictor import AudioPredictor


class _DummyModel:
    def parameters(self):
        yield torch.zeros(1)


class TestAudioPredictorVADModes(unittest.TestCase):
    def test_resolve_backend_translate_defaults_to_wav2vec2(self):
        predictor = AudioPredictor.__new__(AudioPredictor)
        predictor.timestamps_output = "word"
        predictor.vad_mode = "segment"
        predictor.word_timestamps_backend = "auto"
        predictor.audio_task = "translate"
        predictor.language = "fr"
        predictor.word_alignment_model = None

        backend = predictor._resolve_word_timestamps_backend()
        self.assertEqual(backend, "wav2vec2")

    def test_resolve_backend_transcribe_non_en_without_model_raises(self):
        predictor = AudioPredictor.__new__(AudioPredictor)
        predictor.timestamps_output = "word"
        predictor.vad_mode = "segment"
        predictor.word_timestamps_backend = "auto"
        predictor.audio_task = "transcribe"
        predictor.language = "fr"
        predictor.word_alignment_model = None

        with self.assertRaises(ValueError):
            predictor._resolve_word_timestamps_backend()

    def test_resolve_backend_transcribe_non_en_with_custom_model_passes(self):
        predictor = AudioPredictor.__new__(AudioPredictor)
        predictor.timestamps_output = "word"
        predictor.vad_mode = "segment"
        predictor.word_timestamps_backend = "auto"
        predictor.audio_task = "transcribe"
        predictor.language = "fr"
        predictor.word_alignment_model = "VOXPOPULI_ASR_BASE_10K_FR"

        backend = predictor._resolve_word_timestamps_backend()
        self.assertEqual(backend, "wav2vec2")

    def test_resolve_backend_translate_defaults_to_wav2vec2_for_both(self):
        predictor = AudioPredictor.__new__(AudioPredictor)
        predictor.timestamps_output = "both"
        predictor.vad_mode = "segment"
        predictor.word_timestamps_backend = "auto"
        predictor.audio_task = "translate"
        predictor.language = "fr"
        predictor.word_alignment_model = None

        backend = predictor._resolve_word_timestamps_backend()
        self.assertEqual(backend, "wav2vec2")

    def test_resolve_backend_transcribe_non_en_without_model_raises_for_both(self):
        predictor = AudioPredictor.__new__(AudioPredictor)
        predictor.timestamps_output = "both"
        predictor.vad_mode = "segment"
        predictor.word_timestamps_backend = "auto"
        predictor.audio_task = "transcribe"
        predictor.language = "fr"
        predictor.word_alignment_model = None

        with self.assertRaises(ValueError):
            predictor._resolve_word_timestamps_backend()

    def test_predict_segment_mode_requires_vad_metadata(self):
        predictor = AudioPredictor.__new__(AudioPredictor)
        predictor.data_type = "audio"
        predictor.model = _DummyModel()
        predictor.vad_mode = "segment"
        predictor.timestamps_output = "none"
        predictor.word_timestamps_backend = "auto"
        predictor.audio_task = "transcribe"
        predictor.language = "en"
        predictor.word_alignment_model = None
        predictor.verbose = False
        predictor.report_score = False
        predictor.report_time = False

        infer_iter = [
            (
                {
                    "src_type": "waveform",
                    "src": torch.zeros(16000),
                    "speech_segments": [None],
                    "audio_file": ["dummy.wav"],
                },
                0,
            )
        ]

        with self.assertRaises(ValueError):
            predictor._predict(iter(infer_iter))

    def test_predict_segment_mode_accepts_empty_speech_segments(self):
        predictor = AudioPredictor.__new__(AudioPredictor)
        predictor.data_type = "audio"
        predictor.model = _DummyModel()
        predictor.vad_mode = "segment"
        predictor.timestamps_output = "none"
        predictor.word_timestamps_backend = "auto"
        predictor.audio_task = "transcribe"
        predictor.language = "en"
        predictor.word_alignment_model = None
        predictor.verbose = False
        predictor.report_score = False
        predictor.report_time = False
        predictor._predict_vad_segments = lambda waveform, device, speech_segments, word_backend=None: ([], [])

        infer_iter = [
            (
                {
                    "src_type": "waveform",
                    "src": torch.zeros(16000),
                    "speech_segments": [[]],
                    "audio_file": ["dummy.wav"],
                },
                0,
            )
        ]

        scores, estim, predictions = predictor._predict(iter(infer_iter))
        self.assertEqual(scores, [[0.0]])
        self.assertEqual(estim, [[1.0]])
        self.assertEqual(predictions, [[""]])


if __name__ == "__main__":
    unittest.main()
