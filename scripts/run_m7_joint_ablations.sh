#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "${DEBUG_CUDA:-0}" == "1" ]]; then
  export CUDA_LAUNCH_BLOCKING=1
fi

TARGET="${TARGET:-capability}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
EMBEDDING_DIM="${EMBEDDING_DIM:-512}"
HIDDEN_DIM="${HIDDEN_DIM:-1024}"
CODE_EMBEDDING_DIM="${CODE_EMBEDDING_DIM:-128}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-SkillRQ-M7}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-runs/m7_joint_reranker/${TARGET}}"

COMMON_ARGS=(
  --target "${TARGET}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --learning-rate "${LEARNING_RATE}"
  --embedding-dim "${EMBEDDING_DIM}"
  --hidden-dim "${HIDDEN_DIM}"
  --code-embedding-dim "${CODE_EMBEDDING_DIM}"
  --device "${DEVICE}"
  --swanlab-project "${SWANLAB_PROJECT}"
)

if [[ -n "${DATA_ROOT:-}" ]]; then
  COMMON_ARGS+=(--data-root "${DATA_ROOT}")
fi

python3 -m skillrq m7 joint-train \
  "${COMMON_ARGS[@]}" \
  --output-root "${OUTPUT_ROOT_BASE}/joint_base" \
  --swanlab-run-name "m7-${TARGET}-joint-base"

python3 -m skillrq m7 joint-train \
  "${COMMON_ARGS[@]}" \
  --enable-shared-encoder \
  --output-root "${OUTPUT_ROOT_BASE}/shared_encoder" \
  --swanlab-run-name "m7-${TARGET}-joint-shared-encoder"

python3 -m skillrq m7 joint-train \
  "${COMMON_ARGS[@]}" \
  --enable-soft-code-distribution \
  --soft-code-weight "${SOFT_CODE_WEIGHT:-0.1}" \
  --output-root "${OUTPUT_ROOT_BASE}/soft_code_distribution" \
  --swanlab-run-name "m7-${TARGET}-joint-soft-code"

python3 -m skillrq m7 joint-train \
  "${COMMON_ARGS[@]}" \
  --enable-shared-encoder \
  --enable-soft-code-distribution \
  --soft-code-weight "${SOFT_CODE_WEIGHT:-0.1}" \
  --output-root "${OUTPUT_ROOT_BASE}/shared_encoder_soft_code" \
  --swanlab-run-name "m7-${TARGET}-joint-shared-soft"
