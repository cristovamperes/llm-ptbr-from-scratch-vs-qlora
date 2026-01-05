#!/usr/bin/env bash
set -euo pipefail

# Cohesive Transformer v5 training workflow:
#   1. Dataset preflight (stride-specific).
#   2. Model training with improved logging/metrics.
#   3. Artifacts stored per stage (baseline/main/scale).
#
# Usage:
#   ./scripts/run_v5_pipeline.sh [stage]
#     stage ∈ {all, baseline, main, scale}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TOKENIZER_MODEL="${TOKENIZER_MODEL:-versions/v4-subword-lstm/tokenizer_v4k_bf/spm_v4k_bf.model}"
LOG_ROOT="versions/v5-transformer/logs"
MODEL_ROOT="versions/v5-transformer/models"
MAP_ROOT="versions/v5-transformer/mappings"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_SCRIPT="scripts/llm_transformer_v5.py"
PREFLIGHT_SCRIPT="analysis/validate_dataset.py"
SAMPLES_SCRIPT="analysis/generate_samples.py"

mkdir -p "${LOG_ROOT}" "${MODEL_ROOT}" "${MAP_ROOT}" analysis

run_preflight() {
  local stage="$1"
  local stride="$2"
  local seq_len="$3"
  local batch="$4"
  local max_docs="$5"
  local output_json="analysis/dataset_preflight_${stage}.json"
  local samples_txt="analysis/dataset_preflight_${stage}_samples.txt"

  echo "[PIPELINE] Dataset preflight for stage=${stage} (seq_len=${seq_len}, stride=${stride}, batch=${batch})"
  "${PYTHON_BIN}" "${PREFLIGHT_SCRIPT}" \
    --tokenizer_path "${TOKENIZER_MODEL}" \
    --seq_len "${seq_len}" \
    --stride "${stride}" \
    --batch_size "${batch}" \
    --max_docs "${max_docs}" \
    --report_json "${output_json}" \
    --samples_out "${samples_txt}"
}

run_training() {
  local stage="$1"
  local stride="$2"
  local seq_len="$3"
  local epochs="$4"
  local batch="$5"
  local lr_schedule="$6"
  local warmup="$7"
  local restart_steps="$8"
  local restart_m_mul="$9"
  local d_model="${10}"
  local num_layers="${11}"
  local d_ff="${12}"
  local max_docs="${13}"
  local dropout="${14}"
  local label_smoothing="${15}"
  local fallback_penalty="${16}"
  local initial_checkpoint="${17:-}"
  local keep_checkpoints="${18:-6}"

  local log_json="${LOG_ROOT}/train_v5_${stage}.json"
  local csv_log="${LOG_ROOT}/history_v5_${stage}.csv"
  local tb_dir="${LOG_ROOT}/tensorboard_${stage}"
  local metrics_dir="${LOG_ROOT}/metrics_${stage}"
  local model_path="${MODEL_ROOT}/modelo_v5_${stage}.keras"
  local map_path="${MAP_ROOT}/mapeamentos_v5_${stage}.pkl"
  local train_log="${LOG_ROOT}/train_v5_${stage}.log"
  local checkpoint_dir="${LOG_ROOT}/checkpoints_${stage}"

  echo "[PIPELINE] Training stage=${stage}"
  mkdir -p "${checkpoint_dir}"

  local -a train_cmd=(
    "${PYTHON_BIN}" "${TRAIN_SCRIPT}"
    --train
    --tokenizer_path "${TOKENIZER_MODEL}"
    --max_docs "${max_docs}"
    --min_len 200
    --valid_split 0.1
    --seq_len "${seq_len}"
    --stride "${stride}"
    --batch_size "${batch}"
    --epochs "${epochs}"
    --learning_rate 1e-4
    --warmup_steps "${warmup}"
    --lr_schedule "${lr_schedule}"
    --restart_steps "${restart_steps}"
    --restart_t_mul 2.0
    --restart_m_mul "${restart_m_mul}"
    --weight_decay 0.01
    --dropout "${dropout}"
    --d_model "${d_model}"
    --num_layers "${num_layers}"
    --num_heads 8
    --d_ff "${d_ff}"
    --label_smoothing "${label_smoothing}"
    --fallback_penalty "${fallback_penalty}"
    --gradient_clip_norm 1.0
    --model_output "${model_path}"
    --mappings_output "${map_path}"
    --log_json_output "${log_json}"
    --csv_log_output "${csv_log}"
    --tensorboard_logdir "${tb_dir}"
    --metrics_dir "${metrics_dir}"
    --checkpoint_dir "${checkpoint_dir}"
    --keep_n_checkpoints "${keep_checkpoints}"
  )

  if [[ -n "${initial_checkpoint}" ]]; then
    train_cmd+=(--initial_checkpoint "${initial_checkpoint}")
    echo "[PIPELINE] Warm start from checkpoint: ${initial_checkpoint}"
  fi

  "${train_cmd[@]}" >"${train_log}" 2>&1
  local status=$?
  if [[ ${status} -ne 0 ]]; then
    echo "[PIPELINE] Training stage=${stage} FAILED (see ${train_log})"
    return ${status}
  fi

  echo "[PIPELINE] Training log stored at ${train_log}"

  # Promote latest artifacts to the default "fixed" paths for downstream tooling.
  cp -f "${model_path}" "${MODEL_ROOT}/modelo_v5_fixed.keras"
  cp -f "${map_path}" "${MAP_ROOT}/mapeamentos_v5_fixed.pkl"
  if [[ -f "${map_path%.pkl}.json" ]]; then
    cp -f "${map_path%.pkl}.json" "${MAP_ROOT}/mapeamentos_v5_fixed.json"
  fi
}

generate_samples() {
  local stage="$1"
  local output_json="analysis/artifacts/samples/samples_v5_${stage}.json"
  echo "[PIPELINE] Generating qualitative samples for stage=${stage}"
  MODEL_KEYS="v5_brwac_transformer_fixed" OUTPUT_PATH="${output_json}" "${PYTHON_BIN}" "${SAMPLES_SCRIPT}"
}

latest_checkpoint_path() {
  local stage="$1"
  local checkpoint_dir="${LOG_ROOT}/checkpoints_${stage}"
  if [[ ! -d "${checkpoint_dir}" ]]; then
    return 0
  fi
  find "${checkpoint_dir}" -maxdepth 1 -type f -name 'epoch_*.weights.h5' | sort | tail -n 1
}

STAGE="${1:-all}"

case "${STAGE}" in
  baseline|all)
    run_preflight "baseline" 128 256 32 50000
    run_training "baseline" 128 256 2 32 "cosine" 500 0 0.95 256 4 1024 50000 0.1 0.05 0.0
    ;;
esac

case "${STAGE}" in
  main|all)
    run_preflight "main" 64 256 32 50000
    baseline_ckpt="$(latest_checkpoint_path "baseline")"
    if [[ -z "${baseline_ckpt}" ]]; then
      echo "[PIPELINE] Aviso: nenhum checkpoint do stage baseline encontrado; treino main iniciará do zero."
    else
      echo "[PIPELINE] Usando checkpoint baseline como warm start: ${baseline_ckpt}"
    fi
    run_training "main" 64 256 8 32 "cosine_restart" 1000 4000 0.95 256 4 1024 50000 0.1 0.04 0.0 "${baseline_ckpt}" 6
    ;;
esac

case "${STAGE}" in
  scale|scaleplus2)
    run_preflight "scaleplus2" 32 384 18 80000
    initial_scale_ckpt="${INITIAL_CKPT:-$(latest_checkpoint_path "main")}"
    if [[ -z "${initial_scale_ckpt}" ]]; then
      echo "[PIPELINE] ERRO: nenhum checkpoint para warm start encontrado (stage main)."
      echo "[PIPELINE] Informe via variável INITIAL_CKPT=/caminho/para/epoch_xx.weights.h5 ou execute o stage 'main' antes."
      exit 1
    else
      echo "[PIPELINE] Stage scale utilizará checkpoint: ${initial_scale_ckpt}"
    fi
    run_training "scaleplus2" 32 384 12 18 "cosine_restart" 2500 24000 0.5 512 6 2048 80000 0.15 0.03 0.12 "${initial_scale_ckpt}" 8
    ;;
esac

if [[ "${STAGE}" == "all" ]]; then
    generate_samples "pipeline"
fi

echo "[PIPELINE] Completed stage=${STAGE}"
