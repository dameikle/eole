import sys
import types
import unittest
from unittest.mock import patch

import torch

from eole.transforms.silero_vad import SileroVADConfig, SileroVADTransform


class TestSileroVADTransform(unittest.TestCase):
    def _config(self, vad_config=None, sample_rate=16000):
        return types.SimpleNamespace(
            seed=-1,
            transforms_configs=types.SimpleNamespace(silero_vad=vad_config or SileroVADConfig()),
            model=types.SimpleNamespace(encoder=types.SimpleNamespace(sample_rate=sample_rate)),
        )

    def test_warm_up_and_apply_adds_speech_segments(self):
        fake_model = object()

        def fake_load_silero_vad():
            return fake_model

        def fake_get_speech_timestamps(waveform, model, **kwargs):
            self.assertIs(model, fake_model)
            self.assertEqual(kwargs["sampling_rate"], 16000)
            self.assertEqual(kwargs["threshold"], 0.5)
            return [{"start": 1600, "end": 3200}]

        fake_module = types.SimpleNamespace(
            load_silero_vad=fake_load_silero_vad,
            get_speech_timestamps=fake_get_speech_timestamps,
        )

        with patch.dict(sys.modules, {"silero_vad": fake_module}):
            transform = SileroVADTransform(self._config())
            transform.warm_up()
            example = {
                "src": torch.zeros(16000),
                "src_type": "waveform",
                "audio_file": "sample.wav",
            }

            out = transform.apply(example)

        self.assertIs(out, example)
        self.assertEqual(out["speech_segments"], [{"start": 0.1, "end": 0.2}])

    def test_apply_ignores_non_waveform_examples(self):
        transform = SileroVADTransform(self._config())
        example = {"src": "hello world"}

        self.assertIs(transform.apply(example), example)

    def test_warm_up_rejects_unsupported_sample_rate(self):
        transform = SileroVADTransform(self._config(sample_rate=22050))

        with self.assertRaises(ValueError):
            transform.warm_up()


if __name__ == "__main__":
    unittest.main()
