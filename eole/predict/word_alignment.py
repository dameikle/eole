"""Word timestamp alignment backends for audio prediction."""

import logging
import math
import torch

from eole.constants import TORCH_DTYPES

logger = logging.getLogger(__name__)


LANGUAGES_WITHOUT_SPACES = {"ja", "zh"}

TORCHAUDIO_ALIGNMENT_ALIASES = {
    "WAV2VEC2_ASR_BASE_960H": "facebook/wav2vec2-base-960h",
    "VOXPOPULI_ASR_BASE_10K_FR": "facebook/wav2vec2-base-10k-voxpopuli-ft-fr",
    "VOXPOPULI_ASR_BASE_10K_DE": "facebook/wav2vec2-base-10k-voxpopuli-ft-de",
    "VOXPOPULI_ASR_BASE_10K_ES": "facebook/wav2vec2-base-10k-voxpopuli-ft-es",
    "VOXPOPULI_ASR_BASE_10K_IT": "facebook/wav2vec2-base-10k-voxpopuli-ft-it",
}

DEFAULT_ALIGN_MODELS = {
    "en": "facebook/wav2vec2-base-960h",
    "fr": "facebook/wav2vec2-base-10k-voxpopuli-ft-fr",
    "de": "facebook/wav2vec2-base-10k-voxpopuli-ft-de",
    "es": "facebook/wav2vec2-base-10k-voxpopuli-ft-es",
    "it": "facebook/wav2vec2-base-10k-voxpopuli-ft-it",
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "uk": "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "cs": "comodoro/wav2vec2-xls-r-300m-cs-250",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
    "fa": "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
    "el": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    "tr": "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish",
    "da": "saattrupdan/wav2vec2-xls-r-300m-ftspeech",
    "he": "imvladikon/wav2vec2-xls-r-300m-hebrew",
    "vi": "nguyenvulebinh/wav2vec2-base-vi-vlsp2020",
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
    "ur": "kingabzpro/wav2vec2-large-xls-r-300m-Urdu",
    "te": "anuragshas/wav2vec2-large-xlsr-53-telugu",
    "hi": "theainerd/Wav2Vec2-large-xlsr-hindi",
    "ca": "softcatala/wav2vec2-large-xlsr-catala",
    "ml": "gvs/wav2vec2-large-xlsr-malayalam",
    "no": "NbAiLab/nb-wav2vec2-1b-bokmaal-v2",
    "nn": "NbAiLab/nb-wav2vec2-1b-nynorsk",
    "sk": "comodoro/wav2vec2-xls-r-300m-sk-cv8",
    "sl": "anton-l/wav2vec2-large-xlsr-53-slovenian",
    "hr": "classla/wav2vec2-xls-r-parlaspeech-hr",
    "ro": "gigant/romanian-wav2vec2",
    "eu": "stefan-it/wav2vec2-large-xlsr-53-basque",
    "gl": "ifrz/wav2vec2-large-xlsr-galician",
    "ka": "xsway/wav2vec2-large-xlsr-georgian",
    "lv": "jimregan/wav2vec2-large-xlsr-latvian-cv",
    "tl": "Khalsuu/filipino-wav2vec2-l-xls-r-300m-official",
    "sv": "KBLab/wav2vec2-large-voxrex-swedish",
}


def normalize_language_code(language_code):
    if not language_code:
        return "en"
    normalized = str(language_code).strip().lower().replace("_", "-")
    return normalized.split("-", maxsplit=1)[0]


def supports_default_alignment_language(language_code):
    normalized = normalize_language_code(language_code)
    return normalized in DEFAULT_ALIGN_MODELS


def resolve_default_alignment_model(language_code):
    normalized = normalize_language_code(language_code)
    if normalized in DEFAULT_ALIGN_MODELS:
        return DEFAULT_ALIGN_MODELS[normalized], normalized

    supported = sorted(DEFAULT_ALIGN_MODELS)
    raise ValueError(
        "No default wav2vec2 alignment model for language="
        f"'{language_code}' (normalized='{normalized}'). "
        "Set word_alignment_model explicitly or use one of: "
        f"{', '.join(supported)}"
    )


def _normalize_for_ctc(text, language_code):
    no_spaces = normalize_language_code(language_code) in LANGUAGES_WITHOUT_SPACES
    normalized = []
    for ch in text:
        if ch == " " and not no_spaces:
            normalized.append("|")
        elif no_spaces and ch.isspace():
            continue
        else:
            normalized.append(ch.lower())
    return "".join(normalized)


def _make_trellis(emission, tokens, blank_id):
    num_frames, _ = emission.shape
    num_tokens = len(tokens)
    trellis = torch.full((num_frames + 1, num_tokens + 1), -float("inf"), device=emission.device)
    trellis[0, 0] = 0.0
    trellis[1:, 0] = torch.cumsum(emission[:, blank_id], dim=0)

    for t in range(num_frames):
        trellis[t + 1, 1:] = torch.maximum(
            trellis[t, 1:] + emission[t, blank_id],
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


def _backtrack(trellis, emission, tokens, blank_id):
    j = trellis.size(1) - 1
    if j == 0:
        return []

    t = int(torch.argmax(trellis[:, j]).item())

    path = []
    while t > 0 and j > 0:
        stay = trellis[t - 1, j] + emission[t - 1, blank_id]
        change = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]
        t -= 1
        if change > stay:
            j -= 1
            path.append((t, j))

    if j != 0:
        return []
    path.reverse()
    return path


def _interpolate_missing_word_times(words, seg_start, seg_end):
    if not words:
        return words

    default_dur = max((seg_end - seg_start) / max(len(words), 1), 0.02)
    known = [i for i, word in enumerate(words) if word.get("start") is not None and word.get("end") is not None]

    if not known:
        step = max(seg_end - seg_start, 1e-6) / len(words)
        for i, word in enumerate(words):
            word["start"] = seg_start + i * step
            word["end"] = seg_start + (i + 1) * step
        return words

    for i, word in enumerate(words):
        if word.get("start") is not None and word.get("end") is not None:
            continue

        prev_idx = max((k for k in known if k < i), default=None)
        next_idx = min((k for k in known if k > i), default=None)

        if prev_idx is not None and next_idx is not None:
            left = float(words[prev_idx]["end"])
            right = float(words[next_idx]["start"])
            span = max(right - left, 0.0)
            slots = next_idx - prev_idx
            start = left + span * ((i - prev_idx - 1) / slots)
            end = left + span * ((i - prev_idx) / slots)
        elif prev_idx is not None:
            start = float(words[prev_idx]["end"])
            end = min(seg_end, start + default_dur)
        elif next_idx is not None:
            end = float(words[next_idx]["start"])
            start = max(seg_start, end - default_dur)
        else:
            start = seg_start
            end = min(seg_end, seg_start + default_dur)

        word["start"] = start
        word["end"] = end

    return words


def _stabilize_word_times(words, seg_start, seg_end, min_word_dur=0.02):
    if not words:
        return words

    prev_end = seg_start
    for word in words:
        start = float(word.get("start", seg_start))
        end = float(word.get("end", start))

        start = max(seg_start, min(seg_end, start), prev_end)
        end = max(start, min(seg_end, end))

        if end - start < min_word_dur:
            end = min(seg_end, start + min_word_dur)
            if end - start < min_word_dur:
                start = max(seg_start, end - min_word_dur)

        word["start"] = start
        word["end"] = max(start, end)
        prev_end = word["end"]

    for word in words:
        word["start"] = round(max(seg_start, min(seg_end, word["start"])), 2)
        word["end"] = round(max(word["start"], min(seg_end, word["end"])), 2)

    return words


def _cap_outlier_durations(words, seg_start, seg_end, max_word_dur=1.5):
    if not words:
        return words

    for word in words:
        start = float(word["start"])
        end = float(word["end"])
        dur = end - start
        if dur <= max_word_dur:
            continue

        center = 0.5 * (start + end)
        half = 0.5 * max_word_dur
        start = max(seg_start, center - half)
        end = min(seg_end, center + half)
        word["start"] = start
        word["end"] = end

    return words


class Wav2Vec2WordAligner:
    """Multilingual wav2vec2 forced alignment backend."""

    def __init__(
        self,
        model_name=None,
        language="en",
        device="cpu",
        dtype=torch.float32,
        min_word_duration=0.02,
        max_word_duration=1.5,
        enable_outlier_cap=True,
        model_cache_dir=None,
        model_cache_only=False,
    ):
        self.model_name = model_name
        self.language = normalize_language_code(language)
        self.device = device
        self.dtype = dtype
        self.min_word_duration = min_word_duration
        self.max_word_duration = max_word_duration
        self.enable_outlier_cap = enable_outlier_cap
        self.model_cache_dir = model_cache_dir
        self.model_cache_only = model_cache_only
        self.model_dtype = torch.float32
        self.sample_rate = None
        self.model = None
        self.labels = None
        self.label_to_idx = None
        self.blank_id = 0

    def _ensure_loaded(self):
        if self.model is not None:
            return

        if self.model_name is None:
            model_name, normalized_lang = resolve_default_alignment_model(self.language)
        else:
            model_name = TORCHAUDIO_ALIGNMENT_ALIASES.get(self.model_name, self.model_name)
            normalized_lang = self.language

        self.model_dtype = self._resolve_model_dtype()
        self.language = normalized_lang
        self.model_name = model_name

        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        except ImportError as exc:
            raise ImportError(
                "Hugging Face wav2vec2 alignment models require transformers. "
                "Install extras with: pip install eole[align]"
            ) from exc

        processor = Wav2Vec2Processor.from_pretrained(
            model_name,
            cache_dir=self.model_cache_dir,
            local_files_only=self.model_cache_only,
        )
        self.model = Wav2Vec2ForCTC.from_pretrained(
            model_name,
            cache_dir=self.model_cache_dir,
            local_files_only=self.model_cache_only,
        ).to(device=self.device, dtype=self.model_dtype)
        self.model.eval()
        self.labels = processor.tokenizer.get_vocab()
        self.label_to_idx = {label.lower(): idx for label, idx in self.labels.items()}
        self.sample_rate = int(getattr(processor.feature_extractor, "sampling_rate", 16000))

        pad_id = getattr(processor.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            self.blank_id = int(pad_id)
            return
        for pad_label in ("[pad]", "<pad>"):
            if pad_label in self.label_to_idx:
                self.blank_id = int(self.label_to_idx[pad_label])
                return
        self.blank_id = 0

    def _resolve_model_dtype(self):
        requested_dtype = self.dtype
        if isinstance(requested_dtype, str):
            normalized_dtype = requested_dtype.lower()
            if normalized_dtype not in TORCH_DTYPES:
                raise ValueError(f"Invalid wav2vec_dtype value: {requested_dtype}")
            requested_dtype = TORCH_DTYPES[normalized_dtype]

        if requested_dtype == torch.int8:
            logger.warning("wav2vec_dtype=int8 is unstable for alignment; falling back to fp32")
            return torch.float32

        if requested_dtype == torch.float16:
            if self.device.startswith("cuda") or self.device.startswith("mps"):
                return torch.float16
            logger.warning(
                "wav2vec_dtype=fp16 is unsupported on device=%s; falling back to fp32",
                self.device,
            )
            return torch.float32

        if requested_dtype == torch.bfloat16:
            if self.device.startswith("cuda") or self.device.startswith("mps"):
                return torch.bfloat16
            logger.warning(
                "wav2vec_dtype=bf16 is unsupported on device=%s; falling back to fp32",
                self.device,
            )
            return torch.float32

        if requested_dtype != torch.float32:
            raise ValueError(f"Unsupported wav2vec_dtype={requested_dtype}; use fp32, fp16, or bf16.")

        return torch.float32

    def _align_segment_words(self, audio, text, seg_start, seg_end):
        text = text.strip()
        if not text:
            return []

        if self.language in LANGUAGES_WITHOUT_SPACES:
            words = [w for w in text if not w.isspace()]
        else:
            words = [w for w in text.split(" ") if w]
        if not words:
            return []

        if self.label_to_idx is None or self.model is None:
            raise RuntimeError("Alignment model is not loaded.")

        ctc_text = _normalize_for_ctc(text, self.language)
        align_chars = []
        char_map = []
        for idx, ch in enumerate(ctc_text):
            if ch in {" ", "|"} and self.language in LANGUAGES_WITHOUT_SPACES:
                continue
            align_chars.append(ch)
            char_map.append(idx)

        if not align_chars:
            return self._uniform_words(words, seg_start, seg_end)

        with torch.inference_mode():
            emissions = self.model(audio.unsqueeze(0).to(device=self.device, dtype=self.model_dtype)).logits
            emissions = torch.log_softmax(emissions[0], dim=-1)

        has_wildcard = any(ch not in self.label_to_idx for ch in align_chars)
        if has_wildcard:
            non_blank_mask = torch.ones(emissions.size(1), device=emissions.device, dtype=torch.bool)
            non_blank_mask[self.blank_id] = False
            wildcard_col = emissions[:, non_blank_mask].max(dim=1).values
            emissions = torch.cat([emissions, wildcard_col.unsqueeze(1)], dim=1)
            wildcard_id = emissions.size(1) - 1
            token_ids = [self.label_to_idx.get(ch, wildcard_id) for ch in align_chars]
        else:
            token_ids = [self.label_to_idx[ch] for ch in align_chars]

        trellis = _make_trellis(emissions, token_ids, self.blank_id)
        path = _backtrack(trellis, emissions, token_ids, self.blank_id)
        if not path:
            return self._uniform_words(words, seg_start, seg_end)

        num_frames = emissions.size(0)
        duration = max(seg_end - seg_start, 1e-6)
        frame_to_time = duration / num_frames

        char_times = {}
        for frame_idx, tok_idx in path:
            char_idx = char_map[tok_idx]
            if char_idx not in char_times:
                char_times[char_idx] = [frame_idx, frame_idx]
            else:
                char_times[char_idx][1] = frame_idx

        results = []
        cursor = 0
        for word in words:
            start_i = cursor
            end_i = cursor + len(word) - 1
            cursor += len(word)
            if self.language not in LANGUAGES_WITHOUT_SPACES:
                cursor += 1

            available = [char_times[i] for i in range(start_i, end_i + 1) if i in char_times]
            if not available:
                results.append({"text": word, "start": None, "end": None})
                continue

            start_frame = min(v[0] for v in available)
            end_frame = max(v[1] for v in available) + 1
            w_start = seg_start + start_frame * frame_to_time
            w_end = seg_start + end_frame * frame_to_time
            results.append(
                {
                    "text": word,
                    "start": w_start,
                    "end": min(w_end, seg_end),
                }
            )

        results = _interpolate_missing_word_times(results, seg_start, seg_end)
        if self.enable_outlier_cap:
            results = _cap_outlier_durations(
                results,
                seg_start,
                seg_end,
                max_word_dur=self.max_word_duration,
            )
        results = _stabilize_word_times(
            results,
            seg_start,
            seg_end,
            min_word_dur=self.min_word_duration,
        )

        return results

    @staticmethod
    def _uniform_words(words, start, end):
        if not words:
            return []
        span = max(end - start, 1e-6)
        step = span / len(words)
        aligned = []
        for i, word in enumerate(words):
            w_start = start + i * step
            w_end = start + (i + 1) * step
            aligned.append({"text": word, "start": round(w_start, 2), "end": round(w_end, 2)})
        return aligned

    def align(self, waveform, segments, sample_rate):
        self._ensure_loaded()

        if self.model is None:
            raise RuntimeError("Alignment model is not loaded.")

        if self.sample_rate != sample_rate:
            raise ValueError(f"Alignment model expects sample_rate={self.sample_rate}, got {sample_rate}.")

        all_words = []
        total_samples = waveform.shape[0]
        for segment in segments:
            start = max(float(segment["start"]), 0.0)
            end = max(float(segment["end"]), start)
            start_sample = min(int(math.floor(start * sample_rate)), total_samples)
            end_sample = min(int(math.ceil(end * sample_rate)), total_samples)

            if end_sample <= start_sample:
                continue

            chunk = waveform[start_sample:end_sample]
            words = self._align_segment_words(chunk, segment.get("text", ""), start, end)
            all_words.extend(words)

        return all_words
