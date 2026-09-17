# Training OnePO

## Environment

Use **Linux with NVIDIA GPUs**. Prepare Python, CUDA, PyTorch and SGLang using the [verl installation guide at the pinned commit](https://github.com/verl-project/verl/blob/bf48903d93e4618531d3bbae96551a889007dd8b/docs/start/install.rst), then install OnePO:

```bash
git clone https://github.com/FreedomIntelligence/HuatuoGPT-3.git
cd HuatuoGPT-3
python -m pip install -r requirements.txt
```

Dependencies pin verl to **`bf48903d93e4618531d3bbae96551a889007dd8b`** (`0.10.0.dev`). Use this commit rather than another version with the same development label.

The default launcher targets **8 GPUs**. For a different setup, adjust `CUDA_VISIBLE_DEVICES`, `NGPUS_PER_NODE`, and student/grader parallelism together.

## Data and models

Provide local paths for:

| Variable | Required input |
| --- | --- |
| `MODEL_PATH` | Student model, including a tokenizer with a valid chat template. |
| `TRAIN_FILE` | Training data, such as [OnePO-Medical-20K](https://huggingface.co/datasets/FreedomIntelligence/OnePO-Medical-20K). |
| `VAL_FILE` | Validation data using the same task schema. |
| `REWARD_MODEL_PATH` | [HuatuoGPT-3-Grader-8B](https://huggingface.co/FreedomIntelligence/HuatuoGPT-3-Grader-8B) for open-ended rewards. |

JSON arrays, JSONL and Parquet are supported. Each task uses one of these schemas:

| `type` | Input | Scoring target |
| --- | --- | --- |
| `multiple_choice` | `question`, `options` | `answer_idx` |
| `open_ended` | `prompt` (conversation messages) | `rubrics` (`criterion`, signed `points`) |

Include `teacher_response` for cached teacher guidance. Unused task fields may be `null`. For multiple-choice tasks, the loader adds the instruction to return an answer such as `<answer>A</answer>`.

## Start training

Run from the repository root, using absolute paths:

```bash
MODEL_PATH=/path/to/student_model \
TRAIN_FILE=/path/to/onepo_medical_20K.json \
VAL_FILE=/path/to/val.json \
REWARD_MODEL_PATH=/path/to/HuatuoGPT-3-Grader-8B \
bash OnePO.sh
```

The launcher starts a local SGLang grader and stops it when training exits. By default, the grader uses **DP=4, TP=2** and GPU memory fraction **0.15**. Student rollouts use **0.60**. Adjust `REWARD_MODEL_DP`, `REWARD_MODEL_TP`, `REWARD_MODEL_MEM_FRACTION` and `ROLLOUT_GPU_MEMORY_UTILIZATION` for your hardware. Set `REWARD_MODEL_CHAT_TEMPLATE` only when overriding the grader's bundled template.

To use an existing grader, export these settings before running the training command. `REWARD_MODEL_PATH` can then be omitted:

```bash
export START_REWARD_SERVER=0
export REWARD_MODEL_URL=http://127.0.0.1:30001/v1/chat/completions
export REWARD_MODEL_NAME=your-served-grader-name
```

The endpoint must accept OpenAI-compatible requests without authentication. For multiple-choice-only training **and validation**, set `START_REWARD_SERVER=0`. No grader is needed.

## Common settings

Override defaults with environment variables before `bash OnePO.sh`.

| Variable | Default | Purpose |
| --- | ---: | --- |
| `TRAIN_BATCH_SIZE` | 128 | Training batch size. |
| `PPO_MINI_BATCH_SIZE` | 16 | Actor update mini-batch size. |
| `ROLLOUT_N` | 8 | Responses generated per prompt. |
| `ACTOR_LR` | `2e-6` | Actor learning rate. |
| `PROBABILITY_FLOOR` | 0.1 | OnePO probability floor. |
| `MAX_PROMPT_LENGTH` / `MAX_RESPONSE_LENGTH` | 4000 / 8000 | Token limits. |
| `TOTAL_TRAINING_STEPS` | 500 | Maximum training steps. |
| `TEST_FREQ` / `SAVE_FREQ` | 30 / 60 | Validation and checkpoint intervals. |
| `TEST_STEPS` | `[]` | Additional validation steps. |

Preview the launch arguments without starting training or a server:

```bash
DRY_RUN=1 TRAIN_BATCH_SIZE=64 bash OnePO.sh
```

See [`OnePO.sh`](../OnePO.sh) for all options, including GPU parallelism, dynamic batching and reward shaping.

## Validation and outputs

Validation runs every 30 steps and at the final step by default. Set `TEST_FREQ=0 TEST_STEPS='[50,100,200]'` to evaluate only at selected steps, or `TEST_FREQ=0 TEST_STEPS='[]'` to disable scheduled validation. These scores use the training reward. Benchmark reporting should follow the benchmark's official grading protocol.

Outputs are saved under **`runs/`**, or the directory specified by `OUTPUT_DIR`. Checkpoints contain Hugging Face model weights. Automatic optimizer-state resume is disabled. Console logging is enabled by default. To use SwanLab, install and configure it, then set `LOGGER='[console,swanlab]'`.

## Teacher guidance

Cached `teacher_response` values are used first. A teacher response replaces at most one rollout per prompt, only when its reward exceeds every student response in the group.

For online teacher generation, set `ONLINE_TEACHER_ENABLED=True`, `ONEPO_TEACHER_API_URL`, `ONEPO_TEACHER_MODEL` and `ONEPO_TEACHER_API_KEY`. Keep the API key in the environment. Global teacher stopping, thinking-format requirements and final-answer length penalties are optional and disabled by default.
