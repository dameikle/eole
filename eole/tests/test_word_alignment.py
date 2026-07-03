import unittest
import sys
import types
from unittest.mock import patch

import torch

from eole.predict.word_alignment import (
    TORCHAUDIO_ALIGNMENT_ALIASES,
    Wav2Vec2WordAligner,
    _normalize_for_ctc,
    _cap_outlier_durations,
    _interpolate_missing_word_times,
    _stabilize_word_times,
    normalize_language_code,
    resolve_default_alignment_model,
    supports_default_alignment_language,
)


class TestWordAlignmentPostprocessing(unittest.TestCase):
    def test_normalize_language_code(self):
        self.assertEqual(normalize_language_code("en-US"), "en")
        self.assertEqual(normalize_language_code("fr_CA"), "fr")
        self.assertEqual(normalize_language_code(None), "en")

    def test_supports_default_alignment_language(self):
        self.assertTrue(supports_default_alignment_language("fr"))
        self.assertTrue(supports_default_alignment_language("ja"))
        self.assertFalse(supports_default_alignment_language("sw"))

    def test_resolve_default_alignment_model(self):
        model_name, language = resolve_default_alignment_model("fr")
        self.assertEqual(model_name, "facebook/wav2vec2-base-10k-voxpopuli-ft-fr")
        self.assertEqual(language, "fr")

        model_name, language = resolve_default_alignment_model("zh-CN")
        self.assertEqual(model_name, "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn")
        self.assertEqual(language, "zh")

    def test_resolve_default_alignment_model_unsupported_raises(self):
        with self.assertRaises(ValueError):
            resolve_default_alignment_model("sw")

    def test_torchaudio_alignment_aliases_map_to_huggingface_ids(self):
        self.assertEqual(TORCHAUDIO_ALIGNMENT_ALIASES["WAV2VEC2_ASR_BASE_960H"], "facebook/wav2vec2-base-960h")
        self.assertEqual(
            TORCHAUDIO_ALIGNMENT_ALIASES["VOXPOPULI_ASR_BASE_10K_FR"],
            "facebook/wav2vec2-base-10k-voxpopuli-ft-fr",
        )

    def test_wav2vec2_aligner_loads_hf_model_and_aligns(self):
        calls = []

        class _FakeTokenizer:
            pad_token_id = 0

            def get_vocab(self):
                return {"<pad>": 0, "a": 1, "|": 2}

        class _FakeProcessor:
            tokenizer = _FakeTokenizer()
            feature_extractor = types.SimpleNamespace(sampling_rate=16000)

            @classmethod
            def from_pretrained(cls, model_name, **kwargs):
                calls.append(("processor", model_name, kwargs))
                return cls()

        class _FakeModel:
            @classmethod
            def from_pretrained(cls, model_name, **kwargs):
                calls.append(("model", model_name, kwargs))
                return cls()

            def to(self, device=None, dtype=None):
                self.device = device
                self.dtype = dtype
                return self

            def eval(self):
                return self

            def __call__(self, audio):
                logits = torch.tensor(
                    [[[0.0, 5.0, 0.0], [0.0, 5.0, 0.0]]],
                    dtype=torch.float32,
                    device=audio.device,
                )
                return types.SimpleNamespace(logits=logits)

        fake_transformers = types.SimpleNamespace(
            Wav2Vec2ForCTC=_FakeModel,
            Wav2Vec2Processor=_FakeProcessor,
        )

        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            aligner = Wav2Vec2WordAligner(
                model_name="WAV2VEC2_ASR_BASE_960H",
                language="en",
                device="cpu",
            )
            words = aligner.align(
                torch.zeros(16000),
                [{"start": 0.0, "end": 1.0, "text": "a"}],
                sample_rate=16000,
            )

        self.assertEqual(aligner.model_name, "facebook/wav2vec2-base-960h")
        self.assertEqual(calls[0][1], "facebook/wav2vec2-base-960h")
        self.assertEqual(calls[1][1], "facebook/wav2vec2-base-960h")
        self.assertEqual(words, [{"text": "a", "start": 0.0, "end": 0.5}])

    def test_normalize_for_ctc_without_spaces_language(self):
        self.assertEqual(_normalize_for_ctc("你 好", "zh"), "你好")
        self.assertEqual(_normalize_for_ctc("hello world", "en"), "hello|world")

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
