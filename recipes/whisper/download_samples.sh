#!/bin/bash
# Download sample audio files for Whisper testing.
# Requires: wget, ffmpeg

set -euo pipefail

SAMPLE_DIR="${1:-samples}"
mkdir -p "$SAMPLE_DIR"
echo "Downloading samples..."
wget -q --show-progress -O "$SAMPLE_DIR/gb0.ogg" \
    https://upload.wikimedia.org/wikipedia/commons/2/22/George_W._Bush%27s_weekly_radio_address_%28November_1%2C_2008%29.oga
wget -q --show-progress -O "$SAMPLE_DIR/gb1.ogg" \
    https://upload.wikimedia.org/wikipedia/commons/1/1f/George_W_Bush_Columbia_FINAL.ogg
wget -q --show-progress -O "$SAMPLE_DIR/hp0.ogg" \
    https://upload.wikimedia.org/wikipedia/en/d/d4/En.henryfphillips.ogg
wget -q --show-progress -O "$SAMPLE_DIR/mm1.wav" \
    https://cdn.openai.com/whisper/draft-20220913a/micro-machines.wav
wget -q --show-progress -O "$SAMPLE_DIR/fr0.mp3" \
    https://www.archive.org/download/beautyandthemonster_1502_librivox/beautyandthemonster_01_degenlis_64kb.mp3
wget -q --show-progress -O "$SAMPLE_DIR/de0.ogg" \
    https://upload.wikimedia.org/wikipedia/commons/c/c5/De-0%2C2-Liter-Flasche.ogg
wget -q --show-progress -O "$SAMPLE_DIR/es0.ogg" \
    https://upload.wikimedia.org/wikipedia/commons/0/08/Es-1_of_2_Cuento-de-Hadas-article.ogg
wget -q --show-progress -O "$SAMPLE_DIR/ja0.ogg" \
    https://upload.wikimedia.org/wikipedia/commons/8/8c/Ja-0-rei.ogg
wget -q --show-progress -O "$SAMPLE_DIR/zh0.ogg" \
    "https://commons.wikimedia.org/wiki/Special:FilePath/Zh-2004年夏季奥林匹克运动会.ogg"
echo "Converting to 16kHz mono WAV..."
ffmpeg -loglevel -0 -y -i "$SAMPLE_DIR/gb0.ogg" -ar 16000 -ac 1 -c:a pcm_s16le "$SAMPLE_DIR/gb0.wav"
ffmpeg -loglevel -0 -y -i "$SAMPLE_DIR/gb1.ogg" -ar 16000 -ac 1 -c:a pcm_s16le "$SAMPLE_DIR/gb1.wav"
ffmpeg -loglevel -0 -y -i "$SAMPLE_DIR/hp0.ogg" -ar 16000 -ac 1 -c:a pcm_s16le "$SAMPLE_DIR/hp0.wav"
ffmpeg -loglevel -0 -y -i "$SAMPLE_DIR/mm1.wav" -ar 16000 -ac 1 -c:a pcm_s16le "$SAMPLE_DIR/mm0.wav"
ffmpeg -loglevel -0 -y -i "$SAMPLE_DIR/fr0.mp3" -ar 16000 -ac 1 -c:a pcm_s16le "$SAMPLE_DIR/fr0.wav"
ffmpeg -loglevel -0 -y -i "$SAMPLE_DIR/de0.ogg" -t 20 -ar 16000 -ac 1 -c:a pcm_s16le "$SAMPLE_DIR/de0.wav"
ffmpeg -loglevel -0 -y -i "$SAMPLE_DIR/es0.ogg" -t 20 -ar 16000 -ac 1 -c:a pcm_s16le "$SAMPLE_DIR/es0.wav"
ffmpeg -loglevel -0 -y -i "$SAMPLE_DIR/ja0.ogg" -t 20 -ar 16000 -ac 1 -c:a pcm_s16le "$SAMPLE_DIR/ja0.wav"
ffmpeg -loglevel -0 -y -i "$SAMPLE_DIR/zh0.ogg" -t 20 -ar 16000 -ac 1 -c:a pcm_s16le "$SAMPLE_DIR/zh0.wav"
rm -f "$SAMPLE_DIR"/*.ogg "$SAMPLE_DIR/mm1.wav" "$SAMPLE_DIR/fr0.mp3"
echo "Done. Samples in $SAMPLE_DIR/"
ls -la "$SAMPLE_DIR"/*.wav
