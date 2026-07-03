# Whisper

> **NOTE:** To make your life easier, run these commands from the recipe directory (here `recipes/whisper`).

## Supported models

Any HuggingFace Whisper checkpoint can be converted:

| Model | HuggingFace ID | Languages |
|-------|----------------|-----------|
| tiny | `openai/whisper-tiny` | Multilingual |
| tiny.en | `openai/whisper-tiny.en` | English only |
| base | `openai/whisper-base` | Multilingual |
| base.en | `openai/whisper-base.en` | English only |
| small | `openai/whisper-small` | Multilingual |
| small.en | `openai/whisper-small.en` | English only |
| medium | `openai/whisper-medium` | Multilingual |
| medium.en | `openai/whisper-medium.en` | English only |
| large-v3 | `openai/whisper-large-v3` | Multilingual |
| large-v3-turbo | `openai/whisper-large-v3-turbo` | Multilingual |

## Prerequisites

Audio decoding requires FFmpeg libraries to be installed on your system:

```bash
apt install ffmpeg          # Debian/Ubuntu
brew install ffmpeg         # macOS
```

## Quick start

### Set environment variables

```
export EOLE_MODEL_DIR=<where_to_store_models>
```

### Download and convert model

```
eole convert HF --model_dir openai/whisper-base.en --output ${EOLE_MODEL_DIR}/whisper-base-en-eole
```

### Download sample audio files (optional)

```
bash download_samples.sh
```

This downloads a few public-domain audio clips and converts them to 16kHz mono WAV.
Requires `wget` and `ffmpeg`.

### Run inference

The included `audio_files.txt` lists the English sample files. After downloading, run:

```
eole predict -config whisper_predict.yaml -model_path ${EOLE_MODEL_DIR}/whisper-base-en-eole -src ./audio_files.txt -output ./transcription.txt
```

Each line in the output corresponds to the audio file on the same line in the input.

## Output modes

### Plain text (default)

```
eole predict -config whisper_predict.yaml -src ./audio_files.txt -output ./text.txt
```

### Segment timestamps

Outputs JSON with start/end times for each segment:

```
eole predict -config whisper_predict.yaml -src ./audio_files.txt -output ./segments.json -timestamps segment
```

### Word timestamps

Outputs JSON with per-word timing using the configured alignment backend:

```
eole predict -config whisper_predict.yaml -src ./audio_files.txt -output ./words.json -timestamps word
```

Note:
- `word_timestamps_backend: whisper_attn` requires a model with `alignment_heads` in its `generation_config.json` (e.g. `whisper-base.en`, `whisper-small`, `whisper-large-v3`).
- `word_timestamps_backend: wav2vec2` uses Hugging Face wav2vec2 CTC forced alignment and does not depend on Whisper `alignment_heads`.

### Segment and Word Timestamps

Outputs a JSON object containing both segment and word timings:

```
eole predict -config whisper_predict_vad_segment_both.yaml -src ./audio_files.txt -output ./both.json
```

This is useful for downstream subtitle and TTS pipelines that need both segment boundaries and word-level timing.

## VAD modes

EOLE supports optional VAD-aware decoding for audio inference through the standard transform pipeline.

### `vad_mode: none` (default)

- No VAD-aware seeking behavior.
- Uses standard timestamp-seeking decode.

### `vad_mode: chunk_skip`

- Requires `transforms: [silero_vad]`.
- Keeps standard 30s timestamp-seeking decode.
- Skips chunks that have no overlap with detected speech segments.

### `vad_mode: segment`

- Requires `transforms: [silero_vad]`.
- Decodes merged VAD speech regions directly.
- Uses Hugging Face wav2vec2 alignment by default for `timestamps: word` and `timestamps: both`.

Examples:

```bash
eole predict -config whisper_predict_vad.yaml
eole predict -config whisper_predict_vad_chunk_skip.yaml
eole predict -config whisper_predict_vad_segment.yaml
eole predict -config whisper_predict_vad_segment_word.yaml
eole predict -config whisper_predict_vad_segment_both.yaml
```

## Word timestamp backends

Use `word_timestamps_backend` when `timestamps` is `word` or `both`:

- `whisper_attn`: Whisper cross-attention DTW backend.
- `wav2vec2`: Hugging Face wav2vec2 CTC forced alignment backend.
- `auto`: selects backend automatically:
  - `vad_mode: segment` -> `wav2vec2`
  - other modes -> `whisper_attn`

Legacy TorchAudio bundle names such as `WAV2VEC2_ASR_BASE_960H` and `VOXPOPULI_ASR_BASE_10K_FR` are accepted as aliases for equivalent Hugging Face model IDs.

Optional dependencies:

```bash
pip install -e .[vad]          # Silero VAD transform
pip install -e .[align]        # Hugging Face wav2vec2 word alignment
pip install -e .[vad,align]    # VAD segment mode with word timestamps
```

## Language and task

Multilingual Whisper models support specifying the source language and task.

### Language hint

Force a specific source language (useful when auto-detection is unreliable):

```
eole predict -config whisper_predict.yaml -src ./audio_files.txt -language fr
```

### Translation (to English)

Translation requires a multilingual model (not `.en` variants). First convert one:

```
eole convert HF --model_dir openai/whisper-small --output ${EOLE_MODEL_DIR}/whisper-small-eole
```

Then run with `task: translate` on a French audio sample (included in `download_samples.sh`):

```
echo "samples/fr0.wav" > ./french_audio.txt
eole predict -config whisper_predict.yaml -model_path ${EOLE_MODEL_DIR}/whisper-small-eole -src ./french_audio.txt -output ./translation.txt -language fr -task translate
```

Or with segment timestamps:

```
eole predict -config whisper_predict_translate.yaml -model_path ${EOLE_MODEL_DIR}/whisper-small-eole -src ./french_audio.txt -output ./translation.json
```

## Prompt conditioning

Use `initial_prompt` to condition the decoder output style and vocabulary.
The prompt text is prepended as previous context using the `<|startofprev|>` mechanism:

```
eole predict -config whisper_predict.yaml -src ./audio_files.txt -initial_prompt "This is a presidential radio address."
```

This can help with:
- Spelling of proper nouns and domain-specific terms
- Output formatting and punctuation style
- Steering the model toward a particular topic or register

## Evaluation (WER)

### WER scorer plugin

Eole includes a `WerScorer` plugin for computing Word Error Rate during training evaluation. It works like the existing BLEU and TER scorers:

```python
scoring_metrics: ["WER"]
```

Requires the `wer` extra:

```bash
pip install eole[wer]           # published install
pip install -e .[wer]           # local/dev install
```

### LibriSpeech test-clean benchmark

To measure WER on the standard LibriSpeech test-clean dataset (~2620 utterances, ~5.4 hours):

#### 1. Download dataset

```bash
cd recipes/whisper/eval
bash download_librispeech.sh
```

This downloads and extracts `test-clean.tar.gz` (~350MB) from OpenSLR.

#### 2. Run evaluation

```bash
python eval_librispeech.py --model_path $EOLE_MODEL_DIR/whisper-base-en-eole
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--model_path` | required | Path to converted eole model |
| `--data_dir` | `./LibriSpeech/test-clean` | LibriSpeech directory |
| `--gpu` | 0 | GPU device ID (-1 for CPU) |
| `--beam_size` | 5 | Beam search width |
| `--condition_on_previous_text` | false | Condition on previous segment |
| `--output_dir` | `./results` | Where to write results |

The script normalises text using `EnglishTextNormalizer` from the `whisper-normalizer` package (same normaliser used by OpenAI and whisper.cpp) before computing WER via `jiwer`.

## Config reference

| Field | Type | Default  | Description                                                     |
|-------|------|----------|-----------------------------------------------------------------|
| `model_path` | str  | required | Path to converted eole model                                    |
| `src` | str  | required | File listing audio paths (one per line)                         |
| `output` | str  | required | Output file path                                                |
| `beam_size` | int  | 5        | Beam search width                                               |
| `length_penalty` | str  | "avg"    | Length penalty strategy. Use `none` for Whisper models           |
| `max_length` | int  | 250      | Maximum output tokens                                           |
| `batch_size` | int  | 1        | Batch size (use 1 for audio)                                    |
| `gpu_ranks` | list | []       | GPU device IDs                                                  |
| `seed` | int  | -1       | Random seed. Set to 0+ for deterministic fallback sampling      |
| `timestamps` | str  | "none"   | Output mode: "none", "segment", "word", "both"                          |
| `vad_mode` | str | "none" | VAD behavior: "none", "chunk_skip", "segment"                                |
| `word_timestamps_backend` | str | "auto" | Word timestamp backend: "auto", "whisper_attn", "wav2vec2"                   |
| `word_alignment_model` | str | null | Optional Hugging Face wav2vec2 model ID or legacy TorchAudio bundle alias       |
| `word_alignment_min_duration` | float | 0.02 | Minimum per-word duration after wav2vec2 post-processing                     |
| `word_alignment_max_duration` | float | 1.5 | Maximum per-word duration after wav2vec2 post-processing                     |
| `word_alignment_cap_outliers` | bool | true | Clamp long wav2vec2 word-duration outliers                                   |
| `wav2vec_dtype` | str | "fp32" | Compute dtype for wav2vec2 alignment: "fp32", "fp16", "bf16"              |
| `fallback_top_k` | int | 0 | Sampling top-k used during audio fallback steps with temperature > 0             |
| `fallback_top_p` | float | 1.0 | Sampling top-p used during audio fallback steps with temperature > 0           |
| `language` | str  | null     | Source language code (e.g. "en", "fr", "zh")                    |
| `task` | str  | null     | "transcribe" or "translate"                                     |
| `initial_prompt` | str  | null     | Text prompt for decoder conditioning                            |
| `condition_on_previous_text` | bool | false    | Condition next segment on previous segment's text               |
| `fallback_temperatures` | list | [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] | Temperature cascade. Beam search at t=0, sampling at t>0. `[0.0]` to disable |
| `compression_ratio_threshold` | float | 2.4 | Retry at next temperature if gzip compression ratio exceeds this |
| `logprob_threshold` | float | -1.0 | Retry at next temperature if avg log probability is below this   |
| `no_speech_threshold` | float | 0.6 | Low logprob only triggers fallback when no-speech prob is also below this |
