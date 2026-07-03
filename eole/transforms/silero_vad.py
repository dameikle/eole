"""Silero VAD transform for audio preprocessing."""

from pydantic import Field
from eole.transforms import register_transform
from eole.constants import TransformType
from eole.utils.logging import logger
from .transform import Transform, TransformConfig


class SileroVADConfig(TransformConfig):
    threshold: float = Field(default=0.5, description="Speech probability threshold.")
    min_speech_duration_ms: int = Field(default=250, description="Minimum speech segment duration in milliseconds.")
    min_silence_duration_ms: int = Field(
        default=100, description="Minimum silence duration to split segments, in milliseconds."
    )
    speech_pad_ms: int = Field(
        default=30, description="Padding added to each side of speech segments, in milliseconds."
    )
    sample_rate: int | None = Field(
        default=None,
        description="Audio sample rate. Must be 8000 or 16000 (Silero constraint). "
        "Defaults to the encoder's sample_rate if not set.",
    )


@register_transform(name="silero_vad")
class SileroVADTransform(Transform):
    """Voice Activity Detection using Silero VAD.

    Adds ``speech_segments`` metadata to audio examples without modifying
    the waveform.  Non-audio examples pass through unchanged.
    """

    config_model = SileroVADConfig
    type = TransformType.Predict

    def _parse_config(self):
        self.threshold = self.config.threshold
        self.min_speech_duration_ms = self.config.min_speech_duration_ms
        self.min_silence_duration_ms = self.config.min_silence_duration_ms
        self.speech_pad_ms = self.config.speech_pad_ms
        self.sample_rate = self.config.sample_rate

    def warm_up(self, vocabs=None):
        super().warm_up(vocabs)

        # Resolve sample rate: prefer encoder config, fall back to explicit setting
        _model = getattr(self.full_config, "model", None)
        _encoder = getattr(_model, "encoder", None)
        encoder_sr = getattr(_encoder, "sample_rate", None)

        if self.sample_rate is not None and encoder_sr is not None and self.sample_rate != encoder_sr:
            raise ValueError(
                f"silero_vad sample_rate ({self.sample_rate}) does not match "
                f"encoder sample_rate ({encoder_sr}). Either remove sample_rate "
                f"from the silero_vad config to use the encoder default, or set "
                f"it to {encoder_sr}."
            )
        if self.sample_rate is None:
            self.sample_rate = encoder_sr if encoder_sr is not None else 16000

        if self.sample_rate not in (8000, 16000):
            raise ValueError(f"Silero VAD only supports sample_rate 8000 or 16000, got {self.sample_rate}")

        try:
            from silero_vad import load_silero_vad, get_speech_timestamps

            self._model = load_silero_vad()
            self._get_speech_timestamps = get_speech_timestamps
        except ImportError:
            raise ImportError(
                "silero-vad is required for the silero_vad transform. " "Install it with: pip install -e .[vad]"
            )

    def apply(self, example, is_train=False, stats=None, **kwargs):
        if example.get("src_type") != "waveform":
            return example

        import torch

        waveform = example["src"]
        if not isinstance(waveform, torch.Tensor):
            return example

        sr = self.sample_rate
        speech_timestamps = self._get_speech_timestamps(
            waveform,
            self._model,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
            sampling_rate=sr,
            return_seconds=False,
        )

        example["speech_segments"] = [{"start": ts["start"] / sr, "end": ts["end"] / sr} for ts in speech_timestamps]

        audio_dur = waveform.shape[0] / sr
        speech_dur = sum(seg["end"] - seg["start"] for seg in example["speech_segments"])
        audio_file = example.get("audio_file", "?")
        logger.info(
            f"VAD: {len(example['speech_segments'])} speech segments "
            f"({speech_dur:.1f}s / {audio_dur:.1f}s) in {audio_file}"
        )

        return example
