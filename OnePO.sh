#!/usr/bin/env bash
# OnePO training launcher. See docs/training.md for environment and data setup.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/runs}"
PYTHON="${PYTHON:-python3}"
DRY_RUN="${DRY_RUN:-0}"
START_REWARD_SERVER="${START_REWARD_SERVER:-1}"
LOGGER="${LOGGER:-[console]}"

# -----------------------------------------------------------------------------
# Paths: local model/data inputs; outputs are written under OUTPUT_DIR
# -----------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-}"
TRAIN_FILE="${TRAIN_FILE:-}"
VAL_FILE="${VAL_FILE:-}"

PROJECT_NAME="${PROJECT_NAME:-OnePO}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-OnePO_Qwen3_8B}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_DIR}/checkpoints/${EXPERIMENT_NAME}}"
LOGS_DIR="${LOGS_DIR:-${OUTPUT_DIR}/logs}"
OUTPUTS_DIR="${OUTPUTS_DIR:-${OUTPUT_DIR}/outputs}"
SWANLAB_DIR="${SWANLAB_DIR:-${OUTPUT_DIR}/swanlog}"
HYDRA_OUTPUT_DIR="${HYDRA_OUTPUT_DIR:-${OUTPUTS_DIR}/hydra/${EXPERIMENT_NAME}}"
# Python multiprocessing also uses AF_UNIX sockets; keep this path short.
TMP_DIR="${TMP_DIR:-/tmp/onepo-tmp-${UID}-$$}"
# Ray requires a short, unique temporary directory for AF_UNIX sockets.
RAY_SOCKET_DIR="${RAY_SOCKET_DIR:-/tmp/onepo-${UID}-$$}"

# -----------------------------------------------------------------------------
# Paper-aligned training parameters
# -----------------------------------------------------------------------------

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
ROLLOUT_N="${ROLLOUT_N:-8}"
ACTOR_LR="${ACTOR_LR:-2e-6}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4000}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8000}"
# One switch for actor training and actor/reference log-prob computation.
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-False}"
# Fixed-mode sequences per GPU per micro-batch; ignored in dynamic mode.
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
# Dynamic-mode token budget per GPU.
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-12000}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-500}"

# -----------------------------------------------------------------------------
# OnePO teacher and retirement
# -----------------------------------------------------------------------------
PROBABILITY_FLOOR="${PROBABILITY_FLOOR:-0.1}"
# -1 disables the fixed step limit; plateau stopping is configured independently below.
TEACHER_MAX_STEPS="${TEACHER_MAX_STEPS:--1}"
# Cached teacher outputs remain enabled; online generation is opt-in.
ONLINE_TEACHER_ENABLED="${ONLINE_TEACHER_ENABLED:-False}"
# The following settings are only used when online Teacher is enabled.
TEACHER_MODEL="${TEACHER_MODEL:-${ONEPO_TEACHER_MODEL:-gpt-5.4}}"
TEACHER_MAX_TOKENS="${TEACHER_MAX_TOKENS:-8192}"
TEACHER_MAX_CONCURRENCY="${TEACHER_MAX_CONCURRENCY:-64}"
TEACHER_TIMEOUT="${TEACHER_TIMEOUT:-180}"
TEACHER_API_URL="${TEACHER_API_URL:-${ONEPO_TEACHER_API_URL:-}}"

# -1 disables plateau stopping; otherwise all rates must reach MIN_RATE with spread <= TOLERANCE.
RETIREMENT_PLATEAU_WINDOW="${RETIREMENT_PLATEAU_WINDOW:--1}"
RETIREMENT_PLATEAU_TOLERANCE="${RETIREMENT_PLATEAU_TOLERANCE:-0.01}"
RETIREMENT_PLATEAU_MIN_RATE="${RETIREMENT_PLATEAU_MIN_RATE:-0.9}"

# -----------------------------------------------------------------------------
# DAPO: Dynamic Sampling and Overlong Reward Shaping
# -----------------------------------------------------------------------------
ENABLE_FILTER_GROUPS="${ENABLE_FILTER_GROUPS:-True}"
FILTER_GROUPS_METRIC="${FILTER_GROUPS_METRIC:-acc}"
MAX_NUM_GEN_BATCHES="${MAX_NUM_GEN_BATCHES:-20}"

ENABLE_OVERLONG_BUFFER="${ENABLE_OVERLONG_BUFFER:-True}"
OVERLONG_BUFFER_LEN="${OVERLONG_BUFFER_LEN:-4000}"
OVERLONG_PENALTY_FACTOR="${OVERLONG_PENALTY_FACTOR:-0.6}"

# -----------------------------------------------------------------------------
# Final-answer penalty (stacks with total-response DAPO overlong penalty)
# -----------------------------------------------------------------------------
ANSWER_EXPECTED_TOKENS="${ANSWER_EXPECTED_TOKENS:-200}"
ANSWER_BUFFER_TOKENS="${ANSWER_BUFFER_TOKENS:-4000}"
# Set to 0 to disable only the final-answer penalty.
ANSWER_PENALTY_FACTOR="${ANSWER_PENALTY_FACTOR:-0.0}"

# -----------------------------------------------------------------------------
# GPU and rollout
# -----------------------------------------------------------------------------
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
# Ulysses sequence parallelism for actor/reference; 1 disables sequence parallelism.
SP_SIZE="${SP_SIZE:-1}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.60}"
INFER_BACKEND="${INFER_BACKEND:-sglang}"
REQUIRE_THINKING="${REQUIRE_THINKING:-false}"

# -----------------------------------------------------------------------------
# Automatic validation and checkpointing
# -----------------------------------------------------------------------------
TEST_FREQ="${TEST_FREQ:-30}"
TEST_STEPS="${TEST_STEPS:-[]}"
SAVE_FREQ="${SAVE_FREQ:-60}"
MAX_ACTOR_CHECKPOINTS="${MAX_ACTOR_CHECKPOINTS:-5}"
LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-20}"

# -----------------------------------------------------------------------------
# Automatic reward-model deployment
# -----------------------------------------------------------------------------
REWARD_MODEL_PORT="${REWARD_MODEL_PORT:-30001}"
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-}"
REWARD_MODEL_DP="${REWARD_MODEL_DP:-4}"
REWARD_MODEL_TP="${REWARD_MODEL_TP:-2}"
REWARD_MODEL_MEM_FRACTION="${REWARD_MODEL_MEM_FRACTION:-0.15}"
REWARD_MODEL_CHAT_TEMPLATE="${REWARD_MODEL_CHAT_TEMPLATE:-}"
REWARD_MODEL_URL="${REWARD_MODEL_URL:-http://127.0.0.1:${REWARD_MODEL_PORT}/v1/chat/completions}"
REWARD_MODEL_NAME="${REWARD_MODEL_NAME:-${REWARD_MODEL_PATH}}"
REWARD_MODEL_TIMEOUT="${REWARD_MODEL_TIMEOUT:-120}"
AUTO_KILL_REWARD_SERVER="${AUTO_KILL_REWARD_SERVER:-1}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
TRAIN_LOG="${LOGS_DIR}/train_${EXPERIMENT_NAME}_${RUN_ID}.log"
REWARD_MODEL_LOG="${LOGS_DIR}/reward_${EXPERIMENT_NAME}_${RUN_ID}.log"

if [[ "${DRY_RUN}" != "1" ]]; then
  : "${MODEL_PATH:?Set MODEL_PATH to a local model directory (with a chat template)}"
  : "${TRAIN_FILE:?Set TRAIN_FILE to a training JSON, JSONL, or Parquet file}"
  : "${VAL_FILE:?Set VAL_FILE to a validation JSON, JSONL, or Parquet file}"
  for required in "${MODEL_PATH}" "${TRAIN_FILE}" "${VAL_FILE}"; do
    [[ -e "${required}" ]] || { echo "Required path does not exist: ${required}" >&2; exit 2; }
  done
  "${PYTHON}" -c 'import verl; assert verl.__version__ == "0.10.0.dev", "Install the pinned verl commit in requirements.txt"'
  if [[ "${START_REWARD_SERVER}" == "1" ]]; then
    : "${REWARD_MODEL_PATH:?Set REWARD_MODEL_PATH, or START_REWARD_SERVER=0 for an existing grader}"
    [[ -e "${REWARD_MODEL_PATH}" ]] || { echo "Reward model path does not exist" >&2; exit 2; }
    if [[ -n "${REWARD_MODEL_CHAT_TEMPLATE}" && ! -f "${REWARD_MODEL_CHAT_TEMPLATE}" ]]; then
      echo "Reward model chat template does not exist" >&2; exit 2
    fi
  fi
  mkdir -p "${CHECKPOINT_DIR}" "${LOGS_DIR}" "${HYDRA_OUTPUT_DIR}" "${SWANLAB_DIR}" "${TMP_DIR}"
  mkdir -m 700 "${RAY_SOCKET_DIR}"
  exec > >(tee -a "${TRAIN_LOG}") 2>&1
fi

ulimit -n 65535 2>/dev/null || true
ulimit -u 65535 2>/dev/null || true

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export ONEPO_PROBABILITY_FLOOR="${PROBABILITY_FLOOR}"
export REQUIRE_THINKING
export ANSWER_EXPECTED_TOKENS ANSWER_BUFFER_TOKENS ANSWER_PENALTY_FACTOR
export ONEPO_TOKENIZER_PATH="${MODEL_PATH}"
export REWARD_MODEL_URL
export REWARD_MODEL_NAME
export REWARD_MODEL_TIMEOUT
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG="WARN"
export VLLM_LOGGING_LEVEL="WARN"
export VERL_LOGGING_LEVEL="INFO"
export RAY_TMPDIR="${RAY_SOCKET_DIR}"
export TMPDIR="${TMP_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export SWANLAB_LOG_DIR="${SWANLAB_DIR}"

REWARD_SERVER_PID=""

cleanup_reward_server() {
  local exit_code=$?
  if [[ "${AUTO_KILL_REWARD_SERVER}" == "1" && -n "${REWARD_SERVER_PID}" ]]; then
    echo "Stopping reward model server (PID ${REWARD_SERVER_PID})"
    kill -TERM -- "-${REWARD_SERVER_PID}" 2>/dev/null || true
  fi
  return "${exit_code}"
}

start_reward_server() {
  if curl --fail --silent "http://127.0.0.1:${REWARD_MODEL_PORT}/health" >/dev/null; then
    echo "Port ${REWARD_MODEL_PORT} already has a healthy server; refusing to reuse an unverified model." >&2
    return 1
  fi

  echo "Starting reward model: ${REWARD_MODEL_PATH}"
  local template_args=()
  if [[ -n "${REWARD_MODEL_CHAT_TEMPLATE}" ]]; then
    template_args=(--chat-template "${REWARD_MODEL_CHAT_TEMPLATE}")
  fi
  setsid "${PYTHON}" -m sglang.launch_server \
    --model-path "${REWARD_MODEL_PATH}" \
    --port "${REWARD_MODEL_PORT}" \
    --data-parallel-size "${REWARD_MODEL_DP}" \
    --tensor-parallel-size "${REWARD_MODEL_TP}" \
    --mem-fraction-static "${REWARD_MODEL_MEM_FRACTION}" \
    --trust-remote-code \
    "${template_args[@]}" \
    > "${REWARD_MODEL_LOG}" 2>&1 &
  REWARD_SERVER_PID=$!
}

wait_for_reward_server() {
  local attempt
  for attempt in $(seq 1 180); do
    if curl --fail --silent "http://127.0.0.1:${REWARD_MODEL_PORT}/health" >/dev/null; then
      echo "Reward model server is ready."
      return 0
    fi
    if ! kill -0 "${REWARD_SERVER_PID}" 2>/dev/null; then
      echo "Reward model server exited before becoming ready." >&2
      tail -100 "${REWARD_MODEL_LOG}" >&2 || true
      return 1
    fi
    sleep 5
  done
  echo "Timed out waiting for reward model server." >&2
  tail -100 "${REWARD_MODEL_LOG}" >&2 || true
  return 1
}

trap cleanup_reward_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

DATA=(
  "data.train_files=${TRAIN_FILE}"
  "data.val_files=${VAL_FILE}"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.gen_batch_size=${TRAIN_BATCH_SIZE}"
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH}"
  "data.filter_overlong_prompts=True"
  "data.filter_overlong_prompts_workers=16"
  "data.dataloader_num_workers=${DATALOADER_NUM_WORKERS}"
  "data.truncation=error"
  "data.return_raw_chat=True"
  "data.custom_cls.path=${PROJECT_DIR}/onepo/dataset.py"
  "data.custom_cls.name=OnePODataset"
)

ALGORITHM=(
  "algorithm.adv_estimator=grpo"
  "algorithm.use_kl_in_reward=False"
  "algorithm.norm_adv_by_std_in_grpo=True"
  "algorithm.filter_groups.enable=${ENABLE_FILTER_GROUPS}"
  "algorithm.filter_groups.metric=${FILTER_GROUPS_METRIC}"
  "algorithm.filter_groups.max_num_gen_batches=${MAX_NUM_GEN_BATCHES}"
)

MODEL=(
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"
  "actor_rollout_ref.model.use_remove_padding=True"
)

ACTOR=(
  "actor_rollout_ref.actor.strategy=fsdp2"
  "actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SP_SIZE}"
  "actor_rollout_ref.actor.optim.lr=${ACTOR_LR}"
  "actor_rollout_ref.actor.optim.lr_warmup_steps=1"
  "actor_rollout_ref.actor.optim.weight_decay=0.1"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
  "actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}"
  "actor_rollout_ref.actor.clip_ratio_low=0.2"
  "actor_rollout_ref.actor.clip_ratio_high=0.28"
  "actor_rollout_ref.actor.clip_ratio_c=10.0"
  "actor_rollout_ref.actor.policy_loss.loss_mode=onepo"
  "actor_rollout_ref.actor.loss_agg_mode=token-mean"
  "actor_rollout_ref.actor.use_kl_loss=False"
  "actor_rollout_ref.actor.kl_loss_coef=0.0"
  "actor_rollout_ref.actor.entropy_coeff=0.0"
  "actor_rollout_ref.actor.fsdp_config.param_offload=True"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True"
)


REF=(
  "actor_rollout_ref.ref.strategy=fsdp2"
  "actor_rollout_ref.ref.ulysses_sequence_parallel_size=${SP_SIZE}"
  "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ}"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
  "actor_rollout_ref.ref.fsdp_config.param_offload=True"
)

ROLLOUT=(
  "actor_rollout_ref.rollout.name=${INFER_BACKEND}"
  "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
  # Avoid piecewise graph recapture OOM when SGLang resumes after actor updates.
  "+actor_rollout_ref.rollout.engine_kwargs.sglang.disable_piecewise_cuda_graph=True"
  "actor_rollout_ref.rollout.temperature=1.0"
  "actor_rollout_ref.rollout.top_p=1.0"
  "actor_rollout_ref.rollout.top_k=-1"
  "actor_rollout_ref.rollout.calculate_log_probs=False"
  "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ}"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}"
  "+actor_rollout_ref.rollout.agent.agent_loop_manager_class=onepo.manager.OnePOAgentLoopManager"
  "actor_rollout_ref.rollout.agent.default_agent_loop=onepo_single_turn"
  "actor_rollout_ref.rollout.agent.agent_loop_config_path=${PROJECT_DIR}/configs/agent_loop.yaml"
)

REWARD=(
  "reward.num_workers=32"
  "reward.custom_reward_function.path=${PROJECT_DIR}/onepo/reward.py"
  "reward.custom_reward_function.name=compute_score"
  "reward.reward_manager.source=importlib"
  "reward.reward_manager.name=OnePORewardManager"
  "reward.reward_manager.module.path=${PROJECT_DIR}/onepo/reward_manager.py"
  "reward.reward_model.enable=False"
  "+reward.reward_kwargs.overlong_buffer_cfg.enable=${ENABLE_OVERLONG_BUFFER}"
  "+reward.reward_kwargs.overlong_buffer_cfg.len=${OVERLONG_BUFFER_LEN}"
  "+reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${OVERLONG_PENALTY_FACTOR}"
  "+reward.reward_kwargs.overlong_buffer_cfg.log=False"
  "+reward.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH}"
)

ONEPO=(
  "+onepo.probability_floor=${PROBABILITY_FLOOR}"
  "+onepo.teacher_enabled=True"
  "+onepo.teacher_max_steps=${TEACHER_MAX_STEPS}"
  "+onepo.seed=42"
  "+onepo.retirement.plateau_window=${RETIREMENT_PLATEAU_WINDOW}"
  "+onepo.retirement.plateau_tolerance=${RETIREMENT_PLATEAU_TOLERANCE}"
  "+onepo.retirement.plateau_min_retirement_rate=${RETIREMENT_PLATEAU_MIN_RATE}"
  "+onepo.teacher.api_url=${TEACHER_API_URL}"
  "+onepo.teacher.online_enabled=${ONLINE_TEACHER_ENABLED}"
  "+onepo.teacher.model=${TEACHER_MODEL}"
  "+onepo.teacher.max_tokens=${TEACHER_MAX_TOKENS}"
  "+onepo.teacher.max_concurrency=${TEACHER_MAX_CONCURRENCY}"
  "+onepo.teacher.timeout=${TEACHER_TIMEOUT}"
  "+onepo.teacher.max_retries=3"
)

EVALUATION=(
  "actor_rollout_ref.rollout.val_kwargs.temperature=0.7"
  "actor_rollout_ref.rollout.val_kwargs.top_p=1.0"
  "actor_rollout_ref.rollout.val_kwargs.top_k=-1"
  "actor_rollout_ref.rollout.val_kwargs.do_sample=True"
  "actor_rollout_ref.rollout.val_kwargs.n=1"
  "trainer.val_before_train=False"
  "trainer.test_freq=${TEST_FREQ}"
  "+trainer.test_steps=${TEST_STEPS}"
  "trainer.log_val_generations=${LOG_VAL_GENERATIONS}"
)

CHECKPOINTING=(
  "actor_rollout_ref.actor.checkpoint.save_contents=['hf_model']"
  "trainer.default_local_dir=${CHECKPOINT_DIR}"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CHECKPOINTS}"
  "trainer.resume_mode=disable"
)

TRAINER=(
  "hydra.run.dir=${HYDRA_OUTPUT_DIR}"
  "+ray_kwargs.ray_init._temp_dir=${RAY_SOCKET_DIR}"
  "trainer.use_v1=False"
  "trainer.balance_batch=True"
  "trainer.logger=${LOGGER}"
  "trainer.project_name=${PROJECT_NAME}"
  "trainer.experiment_name=${EXPERIMENT_NAME}"
  "trainer.n_gpus_per_node=${NGPUS_PER_NODE}"
  "trainer.nnodes=1"
  "trainer.total_epochs=${TOTAL_EPOCHS}"
  "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
)

echo "============================================================"
echo "OnePO production training"
echo "Project:        ${OUTPUT_DIR}"
echo "Code:           ${PROJECT_DIR}"
echo "Model:          ${MODEL_PATH}"
echo "Train data:     ${TRAIN_FILE}"
echo "Validation:     ${VAL_FILE}"
echo "Checkpoints:    ${CHECKPOINT_DIR}"
echo "Training log:   ${TRAIN_LOG}"
echo "Reward log:     ${REWARD_MODEL_LOG}"
echo "============================================================"

cd "${PROJECT_DIR}"
if [[ "${DRY_RUN}" == "1" ]]; then
  RUNNER=(printf '%s\n')
else
  if [[ "${START_REWARD_SERVER}" == "1" ]]; then
    start_reward_server
    wait_for_reward_server
  fi
  RUNNER=("${PYTHON}" -u -m onepo.train)
fi

"${RUNNER[@]}" \
  "${DATA[@]}" \
  "${ALGORITHM[@]}" \
  "${MODEL[@]}" \
  "${ACTOR[@]}" \
  "${REF[@]}" \
  "${ROLLOUT[@]}" \
  "${REWARD[@]}" \
  "${ONEPO[@]}" \
  "${EVALUATION[@]}" \
  "${CHECKPOINTING[@]}" \
  "${TRAINER[@]}" \
  "$@"

if [[ "${DRY_RUN}" != "1" ]]; then
  echo "Training completed successfully."
fi
