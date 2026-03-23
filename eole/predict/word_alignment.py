"""Word timestamp alignment backends for audio prediction."""

import math
import torch


def _normalize_for_ctc(text):
    normalized = []
    for ch in text:
        if ch == " ":
            normalized.append("|")
        else:
            normalized.append(ch.upper())
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
    """English-first wav2vec2 forced alignment backend."""

    DEFAULT_EN_MODEL = "WAV2VEC2_ASR_BASE_960H"

    def __init__(
        self,
        model_name=None,
        device="cpu",
        min_word_duration=0.02,
        max_word_duration=1.5,
        enable_outlier_cap=True,
    ):
        self.model_name = model_name or self.DEFAULT_EN_MODEL
        self.device = device
        self.min_word_duration = min_word_duration
        self.max_word_duration = max_word_duration
        self.enable_outlier_cap = enable_outlier_cap
        self.bundle = None
        self.model = None
        self.labels = None
        self.label_to_idx = None
        self.blank_id = 0

    def _ensure_loaded(self):
        if self.model is not None:
            return
        try:
            import torchaudio
        except ImportError as exc:
            raise ImportError(
                "wav2vec2 alignment requires torchaudio. Install extras with: pip install eole[align]"
            ) from exc

        if self.model_name not in torchaudio.pipelines.__all__:
            raise ValueError(
                "English-first alignment currently supports torchaudio pipeline bundles. "
                f"Unknown bundle '{self.model_name}'."
            )

        self.bundle = torchaudio.pipelines.__dict__[self.model_name]
        self.model = self.bundle.get_model().to(self.device)
        self.model.eval()
        self.labels = self.bundle.get_labels()
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}

    def _align_segment_words(self, audio, text, seg_start, seg_end):
        text = text.strip()
        if not text:
            return []

        words = [w for w in text.split(" ") if w]
        if not words:
            return []

        if self.label_to_idx is None or self.model is None:
            raise RuntimeError("Alignment model is not loaded.")

        ctc_text = _normalize_for_ctc(text)
        token_ids = []
        char_map = []
        for idx, ch in enumerate(ctc_text):
            if ch in self.label_to_idx:
                token_ids.append(self.label_to_idx[ch])
                char_map.append(idx)

        if not token_ids:
            return self._uniform_words(words, seg_start, seg_end)

        with torch.inference_mode():
            emissions, _ = self.model(audio.unsqueeze(0).to(self.device))
            emissions = torch.log_softmax(emissions[0], dim=-1)

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
            cursor += len(word) + 1

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

        if self.bundle is None:
            raise RuntimeError("Alignment bundle is not loaded.")

        if self.bundle.sample_rate != sample_rate:
            raise ValueError(f"Alignment model expects sample_rate={self.bundle.sample_rate}, got {sample_rate}.")

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
