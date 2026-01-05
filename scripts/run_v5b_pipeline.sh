#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

V5B_ROOT="versions/v5b-transformer"
TOKENIZER_STAGE="${V5B_ROOT}/tokenizers"
MODEL_ROOT="${V5B_ROOT}/models"
MAP_ROOT="${V5B_ROOT}/mappings"
LOG_ROOT="${V5B_ROOT}/logs"
ANALYSIS_ROOT="${V5B_ROOT}/analysis"
CONFIG_ROOT="${V5B_ROOT}/configs"

mkdir -p "${TOKENIZER_STAGE}" "${MODEL_ROOT}" "${MAP_ROOT}" "${LOG_ROOT}" "${ANALYSIS_ROOT}" "${CONFIG_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
TRAIN_SCRIPT="scripts/llm_transformer_v5.py"
TOKENIZER_SCRIPT="scripts/sentencepiece_pipeline.py"
DATA_PREP_SCRIPT="analysis/export_corpus_v5b.py"
FRAG_SCRIPT="analysis/token_fragmentation.py"
SAMPLES_SCRIPT="analysis/generate_samples.py"
GUARD_SCRIPT="analysis/sample_guardrails.py"

MAX_FALLBACK_RATIO="${MAX_FALLBACK_RATIO:-0.03}"
MAX_SHORT_RATIO="${MAX_SHORT_RATIO:-0.12}"

TOKENIZER_PRESETS="${TOKENIZER_PRESETS:-12000:unigram_12k 16000:unigram_16k}"
TOKENIZER_TAG_DEFAULT="${TOKENIZER_TAG:-unigram_12k}"
TOKENIZER_TAG_DRYRUN="${TOKENIZER_TAG_DRYRUN:-${TOKENIZER_TAG_DEFAULT}}"
TOKENIZER_TAG_LONG="${TOKENIZER_TAG_LONG:-${TOKENIZER_TAG_DEFAULT}}"

DRYRUN_EPOCHS="${DRYRUN_EPOCHS:-2}"
LONG_EPOCHS="${LONG_EPOCHS:-12}"
DRYRUN_BATCH="${DRYRUN_BATCH:-48}"
LONG_BATCH="${LONG_BATCH:-64}"
DRYRUN_LR="${DRYRUN_LR:-2.5e-4}"
LONG_LR="${LONG_LR:-1.5e-4}"
DRYRUN_FALLBACK_PENALTY="${DRYRUN_FALLBACK_PENALTY:-0.25}"
LONG_FALLBACK_PENALTY="${LONG_FALLBACK_PENALTY:-0.25}"
DRYRUN_PRECISION="${DRYRUN_PRECISION:-mixed_float16}"
LONG_PRECISION="${LONG_PRECISION:-float32}"
DRYRUN_JIT_COMPILE="${DRYRUN_JIT_COMPILE:-false}"
LONG_JIT_COMPILE="${LONG_JIT_COMPILE:-false}"
MODEL_DROPOUT="${MODEL_DROPOUT:-0.05}"

STAGE="${1:-all}"

to_bool() {
  local value="${1:-false}"
  case "${value,,}" in
    1|true|yes|on)
      echo "true"
      ;;
    *)
      echo "false"
      ;;
  esac
}

ensure_corpus() {
  if [[ ! -f "${V5B_ROOT}/corpus_v5b.txt" ]]; then
    echo "[V5B] Exportando corpus limpo"
    ${PYTHON_BIN} "${DATA_PREP_SCRIPT}" --output "${V5B_ROOT}/corpus_v5b.txt" --min-line-chars 80 --min-alpha-ratio 0.6 --end-inline-sep newline
  fi
}

run_tokenizer() {
  local vocab_size="${1}"
  local tag="${2}"
  local character_coverage="${3:-0.9995}"
  local output_dir="${TOKENIZER_STAGE}/${tag}"
  mkdir -p "${output_dir}"

  ensure_corpus
  echo "[V5B] Treinando tokenizer ${tag} (vocab=${vocab_size}, coverage=${character_coverage})"
  ${PYTHON_BIN} "${TOKENIZER_SCRIPT}" train \
    --input-file "${V5B_ROOT}/corpus_v5b.txt" \
    --output-dir "${output_dir}" \
    --model-prefix "spm_v5b" \
    --model-type unigram \
    --vocab-size "${vocab_size}" \
    --character-coverage "${character_coverage}" \
    --byte-fallback \
    --input-sentence-size 2000000 \
    --stats-output "${output_dir}/stats.json"

  ${PYTHON_BIN} "${FRAG_SCRIPT}" --tokenizer "${output_dir}/spm_v5b.model" --report "${output_dir}/fragmentation_report.json"
}

run_training() {
  local stage_tag="${1}"
  local tokenizer_path="${2}"
  local epochs="${3}"
  local batch="${4}"
  local fallback_penalty="${5}"
  local learning_rate="${6:-2.5e-4}"
  local precision="${7:-}"
  local jit_flag="${8:-false}"

  if [[ ! -f "${tokenizer_path}" ]]; then
    echo "[ERRO] Tokenizer nao encontrado: ${tokenizer_path}" >&2
    exit 1
  fi

  local model_path="${MODEL_ROOT}/modelo_${stage_tag}.keras"
  local mapping_path="${MAP_ROOT}/mapeamentos_${stage_tag}.pkl"
  local log_json="${LOG_ROOT}/train_${stage_tag}.json"
  local csv_log="${LOG_ROOT}/history_${stage_tag}.csv"
  local tb_dir="${LOG_ROOT}/tensorboard_${stage_tag}"
  local metrics_dir="${LOG_ROOT}/metrics_${stage_tag}"
  local checkpoints="${LOG_ROOT}/checkpoints_${stage_tag}"
  mkdir -p "${metrics_dir}" "${checkpoints}"

  local -a extra_args=()
  if [[ -n "${precision}" && "${precision}" != "none" ]]; then
    extra_args+=('--precision' "${precision}")
  fi
  if [[ "${jit_flag}" == "true" ]]; then
    extra_args+=('--jit_compile')
  fi

  local precision_label="auto"
  if [[ -n "${precision}" && "${precision}" != "none" ]]; then
    precision_label="${precision}"
  fi
  echo "[V5B] Treinando estagio ${stage_tag} (batch=${batch}, lr=${learning_rate}, precision=${precision_label}, fallback=${fallback_penalty})"

  ${PYTHON_BIN} "${TRAIN_SCRIPT}" \
    --train \
    --tokenizer_path "${tokenizer_path}" \
    --max_docs 50000 \
    --min_len 200 \
    --valid_split 0.1 \
    --seq_len 256 \
    --stride 48 \
    --batch_size "${batch}" \
    --epochs "${epochs}" \
    --learning_rate "${learning_rate}" \
    --warmup_steps 4000 \
    --weight_decay 0.01 \
    --dropout "${MODEL_DROPOUT}" \
    --label_smoothing 0.0 \
    --fallback_penalty "${fallback_penalty}" \
    --gradient_clip_norm 1.0 \
    --d_model 320 \
    --num_layers 6 \
    --num_heads 8 \
    --d_ff 1280 \
    --model_output "${model_path}" \
    --mappings_output "${mapping_path}" \
    --log_json_output "${log_json}" \
    --csv_log_output "${csv_log}" \
    --tensorboard_logdir "${tb_dir}" \
    --metrics_dir "${metrics_dir}" \
    --checkpoint_dir "${checkpoints}" \
    --keep_n_checkpoints 5 \
    "${extra_args[@]}"

  local latest_checkpoint=""
  if compgen -G "${checkpoints}/epoch_*.weights.h5" > /dev/null; then
    latest_checkpoint="$(ls -1t "${checkpoints}"/epoch_*.weights.h5 | head -n 1)"
  fi
  if [[ -n "${latest_checkpoint}" ]]; then
    local model_stem
    model_stem="$(basename "${model_path}" .keras)"
    local weights_default="${MODEL_ROOT}/${model_stem}.weights.h5"
    local weights_checkpoint="${MODEL_ROOT}/${model_stem}_checkpoint.weights.h5"
    cp -f "${latest_checkpoint}" "${weights_default}"
    cp -f "${latest_checkpoint}" "${weights_checkpoint}"
    echo "[V5B] Pesos sincronizados (${latest_checkpoint##*/} -> ${weights_checkpoint##*/})"
  fi

  local samples_path="${ANALYSIS_ROOT}/samples_${stage_tag}.json"
  MODEL_KEYS="v5_brwac_transformer_${stage_tag}" OUTPUT_PATH="${samples_path}" ${PYTHON_BIN} "${SAMPLES_SCRIPT}"

  if [[ -f "${GUARD_SCRIPT}" && -f "${samples_path}" ]]; then
    local report_path="${ANALYSIS_ROOT}/guardrails_${stage_tag}.json"
    echo "[V5B] Avaliando guardrails (${stage_tag})"
    ${PYTHON_BIN} "${GUARD_SCRIPT}" \
      --samples "${samples_path}" \
      --tokenizer "${tokenizer_path}" \
      --model-key "v5_brwac_transformer_${stage_tag}" \
      --report "${report_path}" \
      --max-fallback-ratio "${MAX_FALLBACK_RATIO}" \
      --max-short-piece-ratio "${MAX_SHORT_RATIO}"
  fi
}

run_tokenizer_presets() {
  for preset in ${TOKENIZER_PRESETS}; do
    IFS=':' read -r vocab tag coverage <<< "${preset}"
    coverage="${coverage:-0.9995}"
    run_tokenizer "${vocab}" "${tag}" "${coverage}"
  done
}

case "${STAGE}" in
  tokenizer)
    run_tokenizer_presets
    ;;
  dryrun)
    dryrun_tokenizer="${TOKENIZER_STAGE}/${TOKENIZER_TAG_DRYRUN}/spm_v5b.model"
    run_training "v5b_dryrun" "${dryrun_tokenizer}" "${DRYRUN_EPOCHS}" "${DRYRUN_BATCH}" "${DRYRUN_FALLBACK_PENALTY}" "${DRYRUN_LR}" "${DRYRUN_PRECISION}" "$(to_bool "${DRYRUN_JIT_COMPILE}")"
    ;;
  long)
    long_tokenizer="${TOKENIZER_STAGE}/${TOKENIZER_TAG_LONG}/spm_v5b.model"
    run_training "v5b_long" "${long_tokenizer}" "${LONG_EPOCHS}" "${LONG_BATCH}" "${LONG_FALLBACK_PENALTY}" "${LONG_LR}" "${LONG_PRECISION}" "$(to_bool "${LONG_JIT_COMPILE}")"
    ;;
  all)
    run_tokenizer_presets
    dryrun_tokenizer="${TOKENIZER_STAGE}/${TOKENIZER_TAG_DRYRUN}/spm_v5b.model"
    run_training "v5b_dryrun" "${dryrun_tokenizer}" "${DRYRUN_EPOCHS}" "${DRYRUN_BATCH}" "${DRYRUN_FALLBACK_PENALTY}" "${DRYRUN_LR}" "${DRYRUN_PRECISION}" "$(to_bool "${DRYRUN_JIT_COMPILE}")"
    long_tokenizer="${TOKENIZER_STAGE}/${TOKENIZER_TAG_LONG}/spm_v5b.model"
    run_training "v5b_long" "${long_tokenizer}" "${LONG_EPOCHS}" "${LONG_BATCH}" "${LONG_FALLBACK_PENALTY}" "${LONG_LR}" "${LONG_PRECISION}" "$(to_bool "${LONG_JIT_COMPILE}")"
    ;;
  *)
    echo "[ERRO] Stage desconhecido: ${STAGE}" >&2
    exit 1
    ;;
esac
