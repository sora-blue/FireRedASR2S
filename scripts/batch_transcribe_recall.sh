#!/usr/bin/env bash
set -euo pipefail

# Batch convert .m4a to 16kHz 16-bit mono PCM wav, then run FireRedASR2S ASR system.
#
# Usage:
#   bash scripts/batch_transcribe_recall.sh [input_dir] [outdir]
#
# Defaults:
#   input_dir=/mnt/d/EnglishRoot/Recall
#   outdir=output_recall

INPUT_DIR="${1:-/mnt/d/EnglishRoot/Recall}"
OUTDIR="${2:-output_recall}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PATH="$PWD/fireredasr2s/:$PATH"
export PYTHONPATH="$PWD/:$PYTHONPATH"

WAV_DIR="$OUTDIR/wav16k"
SCP="$OUTDIR/wav.scp"
mkdir -p "$WAV_DIR"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found. Install it first (e.g. sudo apt-get install -y ffmpeg)." >&2
  exit 1
fi

rm -f "$SCP"
touch "$SCP"

echo "Searching: $INPUT_DIR/*.m4a"
found_any=0
while IFS= read -r -d '' src; do
  found_any=1
  base="$(basename "${src%.*}")"
  dst="$WAV_DIR/$base.wav"
  echo "Convert: $src -> $dst"
  ffmpeg -y -hide_banner -loglevel error -i "$src" -ar 16000 -ac 1 -acodec pcm_s16le -f wav "$dst"
  printf "%s %s\n" "$base" "$dst" >>"$SCP"
done < <(find "$INPUT_DIR" -maxdepth 1 -type f \( -iname "*.m4a" \) -print0 | sort -z)

if [[ "$found_any" -eq 0 ]]; then
  echo "No .m4a files found under: $INPUT_DIR" >&2
  exit 1
fi

use_gpu=1
if ! nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not available in this environment; falling back to CPU." >&2
  use_gpu=0
fi

asr_model_dir="$PWD/pretrained_models/FireRedASR2-AED"
vad_model_dir="$PWD/pretrained_models/FireRedVAD/VAD"
lid_model_dir="$PWD/pretrained_models/FireRedLID"
punc_model_dir="$PWD/pretrained_models/FireRedPunc"

echo "Run ASR system (Shandong dialect usually shows up as LID=zh-north): outdir=$OUTDIR"
CUDA_VISIBLE_DEVICES=0 fireredasr2s-cli \
  --wav_scp "$SCP" \
  --outdir "$OUTDIR" \
  --write_textgrid 1 \
  --write_srt 1 \
  --save_segment 0 \
  --asr_type aed \
  --asr_model_dir "$asr_model_dir" \
  --vad_model_dir "$vad_model_dir" \
  --lid_model_dir "$lid_model_dir" \
  --punc_model_dir "$punc_model_dir" \
  --enable_vad 1 \
  --enable_lid 1 \
  --enable_punc 1 \
  --asr_use_gpu "$use_gpu" \
  --vad_use_gpu "$use_gpu" \
  --lid_use_gpu "$use_gpu" \
  --punc_use_gpu "$use_gpu" \
  --asr_use_half 0 \
  --asr_batch_size 16 \
  --punc_batch_size 32 \
  --beam_size 3 \
  --nbest 1 \
  --decode_max_len 0 \
  --softmax_smoothing 1.25 \
  --aed_length_penalty 0.6 \
  --eos_penalty 1.0 \
  --return_timestamp 1

echo "Done. Results:"
echo "  $OUTDIR/result.jsonl"
echo "  $OUTDIR/asr_srt/"
echo "  $OUTDIR/asr_tg/"
