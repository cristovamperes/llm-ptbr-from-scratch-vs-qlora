#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

V6_ROOT="versions/v6-release-candidate"
TOKENIZER_DIR="${V6_ROOT}/tokenizers"
MODEL_DIR="${V6_ROOT}/models"
MAP_DIR="${V6_ROOT}/mappings"
LOG_DIR="${V6_ROOT}/logs"
ANALYSIS_DIR="${V6_ROOT}/analysis"
CONFIG_DIR="${V6_ROOT}/configs"
mkdir -p "${TOKENIZER_DIR}" "${MODEL_DIR}" "${MAP_DIR}" "${LOG_DIR}" "${ANALYSIS_DIR}" "${CONFIG_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
TRAIN_SCRIPT="scripts/llm_transformer_v5.py"
TOKENIZER_SCRIPT="scripts/sentencepiece_pipeline.py"
DATA_PREP_SCRIPT="analysis/export_corpus_v5b.py"
FRAG_SCRIPT="analysis/token_fragmentation.py"
SAMPLES_SCRIPT="analysis/generate_samples.py"
GUARD_SCRIPT="analysis/sample_guardrails.py"

# Tokenizer presets: vocab:tag
VOCAB_PRESETS="${VOCAB_PRESETS:-12000:unigram_12k 16000:unigram_16k}"
TOKENIZER_TAG="${TOKENIZER_TAG:-unigram_12k}"
BYTE_FALLBACK="${BYTE_FALLBACK:-false}"
CHAR_COVERAGE="${CHAR_COVERAGE:-0.9995}"
INPUT_SENTENCE_SIZE="${INPUT_SENTENCE_SIZE:-2000000}"
MAX_DOCS_CORPUS="${MAX_DOCS_CORPUS:-80000}"

# Modelo / treino
TRAIN_MAX_DOCS="${TRAIN_MAX_DOCS:-50000}"
TRAIN_MIN_LEN="${TRAIN_MIN_LEN:-200}"
VALID_SPLIT="${VALID_SPLIT:-0.1}"

SEQ_LEN="${SEQ_LEN:-256}"
STRIDE="${STRIDE:-48}"
PREFLIGHT_EPOCHS="${PREFLIGHT_EPOCHS:-2}"
PREFLIGHT_BATCH="${PREFLIGHT_BATCH:-64}"
PREFLIGHT_LR="${PREFLIGHT_LR:-2.5e-4}"
LONG_EPOCHS="${LONG_EPOCHS:-24}"
LONG_BATCH="${LONG_BATCH:-96}"
LONG_LR="${LONG_LR:-1.5e-4}"
D_MODEL="${D_MODEL:-512}"
NUM_LAYERS="${NUM_LAYERS:-6}"
NUM_HEADS="${NUM_HEADS:-8}"
D_FF="${D_FF:-2048}"
DROPOUT="${DROPOUT:-0.1}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.05}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
FALLBACK_PENALTY="${FALLBACK_PENALTY:-0.0}"
GRADIENT_CLIP_NORM="${GRADIENT_CLIP_NORM:-1.0}"

NO_LOWERCASE="${NO_LOWERCASE:-true}"
KEEP_NUMBERS="${KEEP_NUMBERS:-true}"
KEEP_UPPER_METADATA="${KEEP_UPPER_METADATA:-true}"
END_INLINE_SEP="${END_INLINE_SEP:-newline}"
MIN_LINE_CHARS="${MIN_LINE_CHARS:-80}"
MIN_ALPHA_RATIO="${MIN_ALPHA_RATIO:-0.6}"

LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
RESTART_STEPS="${RESTART_STEPS:-0}"
RESTART_T_MUL="${RESTART_T_MUL:-2.0}"
RESTART_M_MUL="${RESTART_M_MUL:-1.0}"

PREFLIGHT_TAG="${PREFLIGHT_TAG:-v6_preflight}"
LONG_TAG="${LONG_TAG:-v6_long}"

MAX_FALLBACK_RATIO="${MAX_FALLBACK_RATIO:-0.03}"
MAX_SHORT_PIECE_RATIO="${MAX_SHORT_PIECE_RATIO:-0.12}"

PRECISION_PREFLIGHT="${PRECISION_PREFLIGHT:-auto}"
PRECISION_LONG="${PRECISION_LONG:-auto}"
JIT_PREFLIGHT="${JIT_PREFLIGHT:-false}"
JIT_LONG="${JIT_LONG:-false}"
KEEP_N_CHECKPOINTS="${KEEP_N_CHECKPOINTS:-5}"

STAGE="${1:-all}"

to_bool() {
  local value="${1:-false}"
  case "${value,,}" in
    1|true|yes|on) echo "true" ;;
    *) echo "false" ;;
  esac
}

ensure_corpus() {
  local corpus_path="${V6_ROOT}/corpus_v6.txt"
  if [[ -f "${corpus_path}" ]]; then
    echo "[V6] Corpus ja existe em ${corpus_path}" >&2
    echo "${corpus_path}"
    return
  fi
  echo "[V6] Exportando corpus limpo (max_docs=${MAX_DOCS_CORPUS})" >&2
  ${PYTHON_BIN} "${DATA_PREP_SCRIPT}" \
    --output "${corpus_path}" \
    --max-docs "${MAX_DOCS_CORPUS}" \
    --min-len "${TRAIN_MIN_LEN}" \
    --min-line-chars "${MIN_LINE_CHARS}" \
    --min-alpha-ratio "${MIN_ALPHA_RATIO}" \
    --end-inline-sep "${END_INLINE_SEP}" \
    --keep-numbers \
    --keep-upper-metadata \
    1>&2
  echo "${corpus_path}"
}

run_tokenizer() {
  local vocab_size="${1}"
  local tag="${2}"
  local corpus
  corpus="$(ensure_corpus)"
  local out_dir="${TOKENIZER_DIR}/${tag}"
  mkdir -p "${out_dir}"

  local extra_flags=()
  if [[ "$(to_bool "${BYTE_FALLBACK}")" == "true" ]]; then
    extra_flags+=(--byte-fallback)
  fi

  echo "[V6] Treinando tokenizer ${tag} (vocab=${vocab_size})"
  ${PYTHON_BIN} "${TOKENIZER_SCRIPT}" train \
    --input-file "${corpus}" \
    --output-dir "${out_dir}" \
    --model-prefix "spm_v6" \
    --model-type unigram \
    --vocab-size "${vocab_size}" \
    --character-coverage "${CHAR_COVERAGE}" \
    --input-sentence-size "${INPUT_SENTENCE_SIZE}" \
    --meta-output "${out_dir}/meta.json" \
    --stats-output "${out_dir}/stats.json" \
    "${extra_flags[@]}"

  if [[ -f "${FRAG_SCRIPT}" ]]; then
    ${PYTHON_BIN} "${FRAG_SCRIPT}" --tokenizer "${out_dir}/spm_v6.model" --report "${out_dir}/fragmentation_report.json" || true
  fi
}

run_tokenizer_presets() {
  for preset in ${VOCAB_PRESETS}; do
    IFS=':' read -r vocab tag <<< "${preset}"
    run_tokenizer "${vocab}" "${tag}"
  done
}

run_training() {
  local stage_tag="${1}"
  local tokenizer_path="${2}"
  local epochs="${3}"
  local batch="${4}"
  local learning_rate="${5}"
  local precision="${6}"
  local jit_flag="${7}"

  if [[ ! -f "${tokenizer_path}" ]]; then
    echo "[ERRO] Tokenizer nao encontrado: ${tokenizer_path}" >&2
    exit 1
  fi

  local model_path="${MODEL_DIR}/modelo_${stage_tag}.keras"
  local mapping_path="${MAP_DIR}/mapeamentos_${stage_tag}.pkl"
  local log_json="${LOG_DIR}/train_${stage_tag}.json"
  local csv_log="${LOG_DIR}/history_${stage_tag}.csv"
  local tb_dir="${LOG_DIR}/tensorboard_${stage_tag}"
  local metrics_dir="${LOG_DIR}/metrics_${stage_tag}"
  local checkpoints="${LOG_DIR}/checkpoints_${stage_tag}"
  mkdir -p "${metrics_dir}" "${checkpoints}"

  local -a extra_args=()
  if [[ "${precision}" != "auto" && -n "${precision}" ]]; then
    extra_args+=("--precision" "${precision}")
  fi
  if [[ "$(to_bool "${jit_flag}")" == "true" ]]; then
    extra_args+=("--jit_compile")
  fi
  if [[ "$(to_bool "${NO_LOWERCASE}")" == "true" ]]; then
    extra_args+=("--no_lowercase")
  fi
  if [[ "$(to_bool "${KEEP_NUMBERS}")" == "true" ]]; then
    extra_args+=("--keep_numbers")
  fi
  if [[ "$(to_bool "${KEEP_UPPER_METADATA}")" == "true" ]]; then
    extra_args+=("--keep_upper_metadata")
  fi
  if [[ -n "${END_INLINE_SEP}" ]]; then
    extra_args+=("--end_inline_sep" "${END_INLINE_SEP}")
  fi
  if [[ -n "${MIN_LINE_CHARS}" ]]; then
    extra_args+=("--min_line_chars" "${MIN_LINE_CHARS}")
  fi
  if [[ -n "${MIN_ALPHA_RATIO}" ]]; then
    extra_args+=("--min_alpha_ratio" "${MIN_ALPHA_RATIO}")
  fi
  if [[ -n "${LR_SCHEDULE}" ]]; then
    extra_args+=("--lr_schedule" "${LR_SCHEDULE}")
  fi
  if [[ "${LR_SCHEDULE}" == "cosine_restart" ]]; then
    if [[ "${RESTART_STEPS}" -gt 0 ]]; then
      extra_args+=("--restart_steps" "${RESTART_STEPS}")
    fi
    extra_args+=("--restart_t_mul" "${RESTART_T_MUL}" "--restart_m_mul" "${RESTART_M_MUL}")
  fi

  echo "[V6] Treinando ${stage_tag} (batch=${batch}, lr=${learning_rate}, precision=${precision})"
  ${PYTHON_BIN} "${TRAIN_SCRIPT}" \
    --train \
    --tokenizer_path "${tokenizer_path}" \
    --max_docs "${TRAIN_MAX_DOCS}" \
    --min_len "${TRAIN_MIN_LEN}" \
    --valid_split "${VALID_SPLIT}" \
    --seq_len "${SEQ_LEN}" \
    --stride "${STRIDE}" \
    --batch_size "${batch}" \
    --epochs "${epochs}" \
    --learning_rate "${learning_rate}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --dropout "${DROPOUT}" \
    --label_smoothing "${LABEL_SMOOTHING}" \
    --fallback_penalty "${FALLBACK_PENALTY}" \
    --gradient_clip_norm "${GRADIENT_CLIP_NORM}" \
    --d_model "${D_MODEL}" \
    --num_layers "${NUM_LAYERS}" \
    --num_heads "${NUM_HEADS}" \
    --d_ff "${D_FF}" \
    --model_output "${model_path}" \
    --mappings_output "${mapping_path}" \
    --log_json_output "${log_json}" \
    --csv_log_output "${csv_log}" \
    --tensorboard_logdir "${tb_dir}" \
    --metrics_dir "${metrics_dir}" \
    --checkpoint_dir "${checkpoints}" \
    --keep_n_checkpoints "${KEEP_N_CHECKPOINTS}" \
    "${extra_args[@]}"

  # Sincroniza pesos mais recentes
  local latest_checkpoint=""
  if compgen -G "${checkpoints}/epoch_*.weights.h5" > /dev/null; then
    latest_checkpoint="$(ls -1t "${checkpoints}"/epoch_*.weights.h5 | head -n 1)"
  fi
  if [[ -n "${latest_checkpoint}" ]]; then
    local model_stem
    model_stem="$(basename "${model_path}" .keras)"
    local weights_default="${MODEL_DIR}/${model_stem}.weights.h5"
    local weights_checkpoint="${MODEL_DIR}/${model_stem}_checkpoint.weights.h5"
    cp -f "${latest_checkpoint}" "${weights_default}"
    cp -f "${latest_checkpoint}" "${weights_checkpoint}"
    echo "[V6] Pesos sincronizados (${latest_checkpoint##*/} -> ${weights_checkpoint##*/})"
  fi

  # Opcional: samples e guardrails se scripts suportarem este modelo
  if [[ -f "${SAMPLES_SCRIPT}" ]]; then
    local samples_path="${ANALYSIS_DIR}/samples_${stage_tag}.json"
    local model_key="v6_brwac_transformer_${stage_tag}"
    echo "[V6] Gerando samples (${stage_tag})"
    CUSTOM_MODEL_KEY="${model_key}" \
      CUSTOM_MODEL_PATH="${model_path}" \
      CUSTOM_MAPPING_PATH="${mapping_path}" \
      MODEL_KEYS="${model_key}" \
      OUTPUT_PATH="${samples_path}" \
      ${PYTHON_BIN} "${SAMPLES_SCRIPT}" || true
  fi
  if [[ -f "${GUARD_SCRIPT}" && -f "${ANALYSIS_DIR}/samples_${stage_tag}.json" ]]; then
    local report_path="${ANALYSIS_DIR}/guardrails_${stage_tag}.json"
    local model_key="v6_brwac_transformer_${stage_tag}"
    echo "[V6] Avaliando guardrails (${stage_tag})"
    ${PYTHON_BIN} "${GUARD_SCRIPT}" \
      --samples "${ANALYSIS_DIR}/samples_${stage_tag}.json" \
      --tokenizer "${tokenizer_path}" \
      --model-key "${model_key}" \
      --report "${report_path}" \
      --max-fallback-ratio "${MAX_FALLBACK_RATIO}" \
      --max-short-piece-ratio "${MAX_SHORT_PIECE_RATIO}" || true
  fi
}

case "${STAGE}" in
  tokenizer)
    run_tokenizer_presets
    ;;
  preflight)
    pre_tokenizer="${TOKENIZER_DIR}/${TOKENIZER_TAG}/spm_v6.model"
    run_training "${PREFLIGHT_TAG}" "${pre_tokenizer}" "${PREFLIGHT_EPOCHS}" "${PREFLIGHT_BATCH}" "${PREFLIGHT_LR}" "${PRECISION_PREFLIGHT}" "${JIT_PREFLIGHT}"
    ;;
  train_long)
    long_tokenizer="${TOKENIZER_DIR}/${TOKENIZER_TAG}/spm_v6.model"
    run_training "${LONG_TAG}" "${long_tokenizer}" "${LONG_EPOCHS}" "${LONG_BATCH}" "${LONG_LR}" "${PRECISION_LONG}" "${JIT_LONG}"
    ;;
  all)
    run_tokenizer_presets
    pre_tokenizer="${TOKENIZER_DIR}/${TOKENIZER_TAG}/spm_v6.model"
    run_training "${PREFLIGHT_TAG}" "${pre_tokenizer}" "${PREFLIGHT_EPOCHS}" "${PREFLIGHT_BATCH}" "${PREFLIGHT_LR}" "${PRECISION_PREFLIGHT}" "${JIT_PREFLIGHT}"
    long_tokenizer="${TOKENIZER_DIR}/${TOKENIZER_TAG}/spm_v6.model"
    run_training "${LONG_TAG}" "${long_tokenizer}" "${LONG_EPOCHS}" "${LONG_BATCH}" "${LONG_LR}" "${PRECISION_LONG}" "${JIT_LONG}"
    ;;
  *)
    echo "[ERRO] Stage desconhecido: ${STAGE}" >&2
    exit 1
    ;;
esac
