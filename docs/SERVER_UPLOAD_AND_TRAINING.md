# SkillRQ 服务器上传与训练指南

本文档用于把本地 `/Users/sihan/code/SkillRQ` 迁移到新的 GPU 服务器，并在服务器上执行 M4 / M5 / M7 训练、推理、prompt construction、agent simulation 与评估。

当前建议：**不要上传 raw datasets**，直接上传已经生成好的 `data/processed` 训练视图。服务器主要负责 PyTorch 训练和推理。

---

## 1. 需要上传的内容

### 1.1 必传代码与配置

```text
pyproject.toml
README.md
configs/
scripts/
skillrq/
docs/
tests/              # 可选，但建议上传，用于服务器 smoke test
```

说明：

* `skillrq/` 是核心 Python package。
  * `skillrq/prompting/` 负责 code-path-guided LLM prompt construction。
  * `skillrq/agent_sim/` 负责 mock / vLLM tool-call plan simulation 与 evaluation。
* `scripts/run_m7_joint_ablations.sh` 已包含四组 M7 joint 消融训练。
* `docs/` 不是训练必需，但建议上传，方便服务器端查看命令。
* 上传前确认 `skillrq/m7/joint_model.py` 中存在 `MeanTextEncoder` 与 `token_ids.clamp(...)`，这是修复 CUDA embedding 越界的关键代码。

### 1.2 必传 processed 数据

如果要完整跑 M4 / M5 / M7，上传：

```text
data/processed/
```

如果只跑 M7 joint 消融，最小需要：

```text
data/processed/m7/capability/rerank_examples.jsonl
data/processed/m7/capability/query_candidate_pools.jsonl
data/processed/m7/capability/stats.json
```

如果后续要做 M7 推理，还需要 M4 candidates 和 M5/M4 predictions：

```text
data/processed/m4/capability/candidates.jsonl
data/processed/m4/capability/queries.jsonl
runs/m5_residual_selector/predictions/capability/predictions.jsonl   # 训练完 M5 predict 后才会有
```

如果要运行 Code-Aware M7、Prompt Construction 和 Agent Simulation，建议保留以下运行产物：

```text
runs/m4_query_to_code/predictions/soft_multipath/capability_sequence_eval/predictions.jsonl
runs/m5_code_path_planner/predictions/capability_sequence_eval_m4_prior/predictions.jsonl
runs/m7_code_aware_reranker/capability_sequence_eval/
runs/m7_code_aware_reranker/predictions/capability_sequence_eval_m4_prior/
runs/prompt_construction/capability_sequence_eval_m4_prior/
```

如果要训练 M5：

```text
data/processed/m5/capability/residual_examples.jsonl
data/processed/m5/capability/query_residual_plans.jsonl
data/processed/m5/capability/stats.json
```

如果要训练 M4：

```text
data/processed/m4/capability/candidates.jsonl
data/processed/m4/capability/queries.jsonl
data/processed/m4/capability/train_pairs.jsonl
data/processed/m4/capability/stats.json
```

如果要验证 sequence-aware 指标，建议额外上传：

```text
data/processed/m4_sequence_eval/capability/
data/processed/m5_sequence_eval/capability/
data/processed/m7_sequence_eval/capability/
```

这三个目录从 capability train split 中划出带 tool sequence 的 query 作为 `sequence_dev` / `sequence_test`，用于 First-Tool Accuracy、Transition Accuracy 和 Kendall-tau 等顺序指标。

### 1.3 不建议上传的内容

```text
data/raw/
__pycache__/
.pytest_cache/
.DS_Store
runs/                 # 除非需要迁移已有 checkpoint 或 prediction
reports/              # 除非需要迁移已有报告
```

---

## 2. 推荐上传命令

以下命令在本地执行。

先设置服务器变量：
```bash
export SERVER_PORT=25833
export SERVER_HOST=root@i-2.gpushare.com
export SERVER_DIR=/hy-tmp
```

如果新实例 host / port 变化，只改上面三个变量。

### 2.1 创建服务器目录

```bash
ssh -p "${SERVER_PORT}" "${SERVER_HOST}" "mkdir -p ${SERVER_DIR}"
```

### 2.2 上传代码、配置和文档

```bash
cd /Users/sihan/code/SkillRQ

rsync -av \
  -e "ssh -p ${SERVER_PORT}" \
  --exclude "__pycache__/" \
  --exclude ".pytest_cache/" \
  --exclude ".DS_Store" \
  pyproject.toml README.md configs scripts skillrq docs tests \
  "${SERVER_HOST}:${SERVER_DIR}/"
```

### 2.3 上传 processed 数据

完整上传：

```bash
cd /Users/sihan/code/SkillRQ

rsync -av \
  -e "ssh -p ${SERVER_PORT}" \
  data/processed \
  "${SERVER_HOST}:${SERVER_DIR}/data/"
```

只上传 M7 capability joint 训练最小数据：

```bash
cd /Users/sihan/code/SkillRQ

ssh -p "${SERVER_PORT}" "${SERVER_HOST}" "mkdir -p ${SERVER_DIR}/data/processed/m7/capability"

rsync -av \
  -e "ssh -p ${SERVER_PORT}" \
  data/processed/m7/capability/ \
  "${SERVER_HOST}:${SERVER_DIR}/data/processed/m7/capability/"
```

如果只传最小 M7 数据，后续 `joint-predict` 前还需要补传：

```bash
ssh -p "${SERVER_PORT}" "${SERVER_HOST}" "mkdir -p ${SERVER_DIR}/data/processed/m4/capability"

rsync -av \
  -e "ssh -p ${SERVER_PORT}" \
  data/processed/m4/capability/ \
  "${SERVER_HOST}:${SERVER_DIR}/data/processed/m4/capability/"
```

### 2.4 增量上传当前新增模块

如果服务器上已有项目，只同步最新 CLI、M5/M7、prompting 和 agent simulation 代码：

```bash
scp -P "$SERVER_PORT" /Users/sihan/code/SkillRQ/skillrq/cli.py \
  "$SERVER_HOST:$SERVER_DIR/skillrq/cli.py"

scp -P "$SERVER_PORT" /Users/sihan/code/SkillRQ/skillrq/m5/planning.py \
  "$SERVER_HOST:$SERVER_DIR/skillrq/m5/planning.py"

scp -P "$SERVER_PORT" /Users/sihan/code/SkillRQ/skillrq/m7/data.py \
  "$SERVER_HOST:$SERVER_DIR/skillrq/m7/data.py"

scp -P "$SERVER_PORT" /Users/sihan/code/SkillRQ/skillrq/m7/model.py \
  "$SERVER_HOST:$SERVER_DIR/skillrq/m7/model.py"

scp -P "$SERVER_PORT" /Users/sihan/code/SkillRQ/skillrq/m7/train.py \
  "$SERVER_HOST:$SERVER_DIR/skillrq/m7/train.py"

scp -P "$SERVER_PORT" /Users/sihan/code/SkillRQ/skillrq/m7/predict.py \
  "$SERVER_HOST:$SERVER_DIR/skillrq/m7/predict.py"

scp -P "$SERVER_PORT" -r /Users/sihan/code/SkillRQ/skillrq/prompting \
  "$SERVER_HOST:$SERVER_DIR/skillrq/prompting"

scp -P "$SERVER_PORT" -r /Users/sihan/code/SkillRQ/skillrq/agent_sim \
  "$SERVER_HOST:$SERVER_DIR/skillrq/agent_sim"
```

---

## 3. 服务器环境初始化

以下命令在服务器执行。

```bash
cd /hy-tmp
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
uv pip install swanlab
```

安装 PyTorch 时不要安装 `torchvision` / `torchaudio`，本项目训练只需要 `torch`。

建议先尝试与你服务器 CUDA 匹配的 PyTorch wheel。若是 RTX 50 系 / Blackwell，通常需要 CUDA 12.8 或更新的 PyTorch wheel：

```bash
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

如果需要运行真实 LLM / vLLM tool-call plan 推理，额外安装：

```bash
uv pip install -U vllm
python -c "import vllm; print(vllm.__version__)"
```

验证 GPU：

```bash
python3 - <<'PY'
import torch
print("torch =", torch.__version__)
print("cuda available =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device =", torch.cuda.get_device_name(0))
    print("capability =", torch.cuda.get_device_capability(0))
PY
```

验证 CLI：

```bash
python3 -m skillrq --help
python3 -m skillrq m7 joint-train --help
python3 -m skillrq prompt build --help
python3 -m skillrq agent-sim vllm --help
```

验证关键补丁是否已同步：

```bash
python3 - <<'PY'
import inspect
from skillrq.m7.joint_model import build_joint_reranker_model
src = inspect.getsource(build_joint_reranker_model)
print("has MeanTextEncoder =", "MeanTextEncoder" in src)
print("has clamp =", "token_ids.clamp" in src)
print("has EmbeddingBag =", "EmbeddingBag" in src)
PY
```

期望输出：

```text
has MeanTextEncoder = True
has clamp = True
has EmbeddingBag = False
```

---

## 4. 训练顺序

如果服务器只负责 M7 joint 消融，直接看 `4.4`。

### 4.0 可选：构建 sequence-aware 评估视图

如果本地尚未生成 sequence-aware 数据视图，可在服务器上先执行：

```bash
python3 -m skillrq m4 sequence-split \
  --target capability \
  --sequence-dev-size 2000 \
  --sequence-test-size 5000 \
  --seed 13

python3 -m skillrq m5 prepare \
  --target capability \
  --m4-data-root data/processed/m4_sequence_eval/capability \
  --output-root data/processed/m5_sequence_eval/capability \
  --max-steps 6

python3 -m skillrq m7 prepare \
  --target capability \
  --m4-data-root data/processed/m4_sequence_eval/capability \
  --output-root data/processed/m7_sequence_eval/capability \
  --negatives-per-positive 2
```

当前本地已生成的数据规模：

```text
M4 queries: train=321,107 sequence_dev=2,000 sequence_test=5,000 test=1,100
M5 residual_examples: train=778,957 sequence_dev=4,966 sequence_test=12,236 test=2,577
M7 rerank_examples: train=2,379,907 sequence_dev=15,174 sequence_test=37,333 test=7,887
```

### 4.1 M4 Query-to-Code 训练

```bash
cd /hy-tmp
source .venv/bin/activate

python3 -m skillrq m4 train \
  --target capability \
  --epochs 20 \
  --batch-size 2048 \
  --learning-rate 3e-4 \
  --embedding-dim 512 \
  --hidden-dim 1024 \
  --device cuda \
  --swanlab-project SkillRQ-M4 \
  --swanlab-run-name m4-capability-query-to-code
```

sequence-aware 评估视图训练：

```bash
python3 -m skillrq m4 train \
  --target capability \
  --data-root data/processed/m4_sequence_eval/capability \
  --output-root runs/m4_query_to_code/capabilityrq/capability_sequence_eval \
  --epochs 20 \
  --batch-size 2048 \
  --learning-rate 3e-4 \
  --embedding-dim 512 \
  --hidden-dim 1024 \
  --device cuda \
  --swanlab-project SkillRQ-M4 \
  --swanlab-run-name m4-capability-sequence-eval
```

soft multi-path M4 训练：

```bash
python3 -m skillrq m4 train \
  --target capability \
  --model-kind soft-multipath \
  --data-root data/processed/m4/capability \
  --output-root runs/m4_query_to_code/soft_multipath/capability \
  --epochs 20 \
  --batch-size 512 \
  --learning-rate 3e-4 \
  --embedding-dim 512 \
  --hidden-dim 1024 \
  --code-embedding-dim 256 \
  --hierarchy-weight 1.0 \
  --contrastive-weight 1.0 \
  --path-bce-weight 0.2 \
  --contrastive-negative-count 512 \
  --temperature 0.07 \
  --device cuda \
  --swanlab-project SkillRQ-M4 \
  --swanlab-run-name m4-soft-multipath-capability
```

soft multi-path M4 的 sequence-aware 训练：

```bash
python3 -m skillrq m4 train \
  --target capability \
  --model-kind soft-multipath \
  --data-root data/processed/m4_sequence_eval/capability \
  --output-root runs/m4_query_to_code/soft_multipath/capability_sequence_eval \
  --epochs 20 \
  --batch-size 512 \
  --learning-rate 3e-4 \
  --embedding-dim 512 \
  --hidden-dim 1024 \
  --code-embedding-dim 256 \
  --hierarchy-weight 1.0 \
  --contrastive-weight 1.0 \
  --path-bce-weight 0.2 \
  --contrastive-negative-count 512 \
  --temperature 0.07 \
  --device cuda \
  --swanlab-project SkillRQ-M4 \
  --swanlab-run-name m4-soft-multipath-sequence-eval
```

M4 推理：

```bash
python3 -m skillrq m4 predict \
  --target capability \
  --checkpoint-root runs/m4_query_to_code/capabilityrq/capability \
  --output-root runs/m4_query_to_code/predictions/capability \
  --top-n-paths 16 \
  --candidate-budget 100 \
  --split test \
  --device cuda \
  --swanlab-project SkillRQ-M4 \
  --swanlab-run-name m4-capability-predict
```

M4 sequence-test 推理：

```bash
python3 -m skillrq m4 predict \
  --target capability \
  --data-root data/processed/m4_sequence_eval/capability \
  --checkpoint-root runs/m4_query_to_code/capabilityrq/capability_sequence_eval \
  --output-root runs/m4_query_to_code/predictions/capability_sequence_eval \
  --top-n-paths 16 \
  --candidate-budget 100 \
  --split sequence_test \
  --device cuda \
  --swanlab-project SkillRQ-M4 \
  --swanlab-run-name m4-capability-sequence-test-predict
```

soft multi-path M4 推理：

```bash
python3 -m skillrq m4 predict \
  --target capability \
  --model-kind soft-multipath \
  --data-root data/processed/m4/capability \
  --checkpoint-root runs/m4_query_to_code/soft_multipath/capability \
  --output-root runs/m4_query_to_code/predictions/soft_multipath/capability \
  --top-n-paths 16 \
  --candidate-budget 100 \
  --beam-width 8 \
  --score-blend 0.65 \
  --split test \
  --device cuda \
  --swanlab-project SkillRQ-M4 \
  --swanlab-run-name m4-soft-multipath-predict
```

soft multi-path M4 的 sequence-test 推理：

```bash
python3 -m skillrq m4 predict \
  --target capability \
  --model-kind soft-multipath \
  --data-root data/processed/m4_sequence_eval/capability \
  --checkpoint-root runs/m4_query_to_code/soft_multipath/capability_sequence_eval \
  --output-root runs/m4_query_to_code/predictions/soft_multipath/capability_sequence_eval \
  --top-n-paths 16 \
  --candidate-budget 100 \
  --beam-width 8 \
  --score-blend 0.65 \
  --split sequence_test \
  --device cuda \
  --swanlab-project SkillRQ-M4 \
  --swanlab-run-name m4-soft-multipath-sequence-test-predict
```

M4 评估：

```bash
python3 -m skillrq m4 evaluate \
  --prediction-path runs/m4_query_to_code/predictions/capability/predictions.jsonl \
  --output-path reports/tables/m4_capabilityrq_capability_metrics.json \
  --top-k 1,5,10,20,50,100 \
  --set-metric-name tool_set_recall
```

### 4.2 M5 Residual Selector 训练

如果要训练新版 residual code path planner，先基于 soft multi-path M4 predictions 准备 M5 code-plan 数据：

```bash
python3 -m skillrq m5 prepare \
  --target capability \
  --model-kind code-plan \
  --m4-data-root data/processed/m4_sequence_eval/capability \
  --m4-prediction-path runs/m4_query_to_code/predictions/soft_multipath/capability_sequence_eval/predictions.jsonl \
  --output-root data/processed/m5_code_plan/capability_sequence_eval \
  --max-steps 6
```

如果只想先跑 oracle planning smoke test，可以去掉 `--m4-prediction-path`。

```bash
python3 -m skillrq m5 train \
  --target capability \
  --epochs 20 \
  --batch-size 2048 \
  --learning-rate 3e-4 \
  --embedding-dim 512 \
  --hidden-dim 1024 \
  --coverage-weight 1.0 \
  --device cuda \
  --swanlab-project SkillRQ-M5 \
  --swanlab-run-name m5-capability-coverage
```

sequence-aware 评估视图训练：

```bash
python3 -m skillrq m5 train \
  --target capability \
  --data-root data/processed/m5_sequence_eval/capability \
  --output-root runs/m5_residual_selector/capability_sequence_eval \
  --epochs 20 \
  --batch-size 2048 \
  --learning-rate 3e-4 \
  --embedding-dim 512 \
  --hidden-dim 1024 \
  --coverage-weight 1.0 \
  --device cuda \
  --swanlab-project SkillRQ-M5 \
  --swanlab-run-name m5-capability-sequence-eval
```

新版 code path planner 训练：

```bash
python3 -m skillrq m5 train \
  --target capability \
  --model-kind code-plan \
  --data-root data/processed/m5_code_plan/capability_sequence_eval \
  --output-root runs/m5_code_path_planner/capability_sequence_eval \
  --epochs 20 \
  --batch-size 2048 \
  --learning-rate 3e-4 \
  --embedding-dim 512 \
  --hidden-dim 1024 \
  --coverage-weight 1.0 \
  --role-weight 0.3 \
  --stop-weight 0.3 \
  --device cuda \
  --swanlab-project SkillRQ-M5 \
  --swanlab-run-name m5-code-path-planner-sequence-eval
```

M5 推理：

```bash
python3 -m skillrq m5 predict \
  --target capability \
  --checkpoint-root runs/m5_residual_selector/capability \
  --output-root runs/m5_residual_selector/predictions/capability \
  --max-steps 6 \
  --top-n-paths 16 \
  --candidates-per-step 20 \
  --split test \
  --device cuda \
  --swanlab-project SkillRQ-M5 \
  --swanlab-run-name m5-capability-predict
```

M5 sequence-test 推理：

```bash
python3 -m skillrq m5 predict \
  --target capability \
  --m4-data-root data/processed/m4_sequence_eval/capability \
  --checkpoint-root runs/m5_residual_selector/capability_sequence_eval \
  --output-root runs/m5_residual_selector/predictions/capability_sequence_eval \
  --max-steps 6 \
  --top-n-paths 16 \
  --candidates-per-step 20 \
  --split sequence_test \
  --device cuda \
  --swanlab-project SkillRQ-M5 \
  --swanlab-run-name m5-capability-sequence-test-predict
```

新版 code path planner sequence-test 推理：

```bash
python3 -m skillrq m5 predict \
  --target capability \
  --model-kind code-plan \
  --m4-data-root data/processed/m4_sequence_eval/capability \
  --m4-prediction-path runs/m4_query_to_code/predictions/soft_multipath/capability_sequence_eval/predictions.jsonl \
  --checkpoint-root runs/m5_code_path_planner/capability_sequence_eval \
  --output-root runs/m5_code_path_planner/predictions/capability_sequence_eval_m4_prior \
  --max-steps 6 \
  --top-n-paths 16 \
  --candidates-per-step 20 \
  --stop-threshold 0.55 \
  --split sequence_test \
  --device cuda \
  --swanlab-project SkillRQ-M5 \
  --swanlab-run-name m5-code-path-plan-sequence-test
```

说明：

* `code-plan` 推理会复用 M4 soft multi-path 输出中的 `retrieved_capabilities`，并记录 `m4_candidate_reuse_rate`、`m4_hit_rate`、`m5_hit_rate`、`m4_hit_m5_miss_rate`。
* 推理时会构建 candidate retrieval index，避免每个 step 遍历全量 candidates。

如果要启用 Phase 1 的 exact-first bucket retrieval 消融：

```bash
python3 -m skillrq m5 predict \
  --target capability \
  --model-kind code-plan \
  --m4-data-root data/processed/m4_sequence_eval/capability \
  --m4-prediction-path runs/m4_query_to_code/predictions/soft_multipath/capability_sequence_eval/predictions.jsonl \
  --checkpoint-root runs/m5_code_path_planner/capability_sequence_eval \
  --output-root runs/m5_code_path_planner/predictions/capability_sequence_eval_exact_first_m4_prior \
  --max-steps 6 \
  --top-n-paths 16 \
  --candidates-per-step 20 \
  --stop-threshold 0.55 \
  --split sequence_test \
  --enable-exact-first-retrieval \
  --device cuda \
  --swanlab-project SkillRQ-M5 \
  --swanlab-run-name m5-code-path-plan-exact-first-m4-prior
```

若要做 Phase 1 only 消融，可额外加：

```bash
--disable-m4-candidate-prior
```

M5 评估：

```bash
python3 -m skillrq m5 evaluate \
  --prediction-path runs/m5_residual_selector/predictions/capability/predictions.jsonl \
  --output-path reports/tables/m5_coverage_supervision_capability.json \
  --top-k 5,10,20,50,100 \
  --set-metric-name tool_set_recall
```

新版 code path planner 评估：

```bash
python3 -m skillrq m5 evaluate \
  --prediction-path runs/m5_code_path_planner/predictions/capability_sequence_eval_m4_prior/predictions.jsonl \
  --output-path reports/tables/m5_code_plan_m4_prior_sequence_test.json \
  --top-k 5,10,20,50,100 \
  --set-metric-name tool_set_recall
```

M5 evaluate 偏低时，先运行 Phase 0 诊断：

```bash
python3 scripts/diagnose_m5_nested_candidates.py \
  --prediction-path runs/m5_code_path_planner/predictions/capability_sequence_eval_m4_prior/predictions.jsonl \
  --output-path reports/tables/m5_nested_candidates_diagnostic.json

python3 scripts/diagnose_m5_codepath_to_candidate.py \
  --prediction-path runs/m5_code_path_planner/predictions/capability_sequence_eval_m4_prior/predictions.jsonl \
  --m4-data-root data/processed/m4_sequence_eval/capability \
  --output-path reports/tables/m5_codepath_to_candidate_diagnostic.json

python3 scripts/diagnose_m4_m5_candidate_reuse.py \
  --m4-prediction-path runs/m4_query_to_code/predictions/soft_multipath/capability_sequence_eval/predictions.jsonl \
  --m5-prediction-path runs/m5_code_path_planner/predictions/capability_sequence_eval_m4_prior/predictions.jsonl \
  --output-path reports/tables/m4_m5_candidate_reuse_diagnostic.json
```

### 4.3 M7 Offline Reranker 训练

```bash
python3 -m skillrq m7 train \
  --target capability \
  --epochs 10 \
  --batch-size 2048 \
  --learning-rate 3e-4 \
  --embedding-dim 512 \
  --hidden-dim 1024 \
  --role-weight 0.2 \
  --stage-weight 0.2 \
  --order-weight 0.2 \
  --device cuda \
  --swanlab-project SkillRQ-M7 \
  --swanlab-run-name m7-capability-reranker
```

sequence-aware 评估视图训练：

```bash
python3 -m skillrq m7 train \
  --target capability \
  --data-root data/processed/m7_sequence_eval/capability \
  --output-root runs/m7_reranker/capability_sequence_eval \
  --epochs 10 \
  --batch-size 2048 \
  --learning-rate 3e-4 \
  --embedding-dim 512 \
  --hidden-dim 1024 \
  --role-weight 0.2 \
  --stage-weight 0.2 \
  --order-weight 0.2 \
  --device cuda \
  --swanlab-project SkillRQ-M7 \
  --swanlab-run-name m7-capability-sequence-eval
```

Code-Aware M7 训练：

```bash
python3 -m skillrq m7 train \
  --target capability \
  --model-kind code-aware \
  --data-root data/processed/m7_sequence_eval/capability \
  --output-root runs/m7_code_aware_reranker/capability_sequence_eval \
  --epochs 10 \
  --batch-size 2048 \
  --learning-rate 3e-4 \
  --embedding-dim 512 \
  --hidden-dim 1024 \
  --role-weight 0.2 \
  --stage-weight 0.2 \
  --order-weight 0.2 \
  --code-consistency-weight 0.3 \
  --schema-weight 0.2 \
  --coverage-gain-weight 0.2 \
  --prompt-usefulness-weight 0.3 \
  --device cuda \
  --swanlab-project SkillRQ-M7 \
  --swanlab-run-name m7-code-aware-capability-sequence-eval
```

M7 offline 推理，输入建议使用 M5 predictions：

```bash
python3 -m skillrq m7 predict \
  --target capability \
  --prediction-path runs/m5_residual_selector/predictions/capability/predictions.jsonl \
  --checkpoint-root runs/m7_reranker/capability \
  --output-root runs/m7_reranker/predictions/capability \
  --top-k 100 \
  --device cuda \
  --swanlab-project SkillRQ-M7 \
  --swanlab-run-name m7-capability-predict
```

Code-Aware M7 推理，输入建议使用 M5 code-plan m4-prior predictions：

```bash
python3 -m skillrq m7 predict \
  --target capability \
  --model-kind code-aware \
  --m4-data-root data/processed/m4_sequence_eval/capability \
  --prediction-path runs/m5_code_path_planner/predictions/capability_sequence_eval_m4_prior/predictions.jsonl \
  --checkpoint-root runs/m7_code_aware_reranker/capability_sequence_eval \
  --output-root runs/m7_code_aware_reranker/predictions/capability_sequence_eval_m4_prior \
  --top-k 100 \
  --device cuda \
  --swanlab-project SkillRQ-M7 \
  --swanlab-run-name m7-code-aware-capability-predict
```

M7 sequence-test 推理：

```bash
python3 -m skillrq m7 predict \
  --target capability \
  --m4-data-root data/processed/m4_sequence_eval/capability \
  --prediction-path runs/m5_residual_selector/predictions/capability_sequence_eval/predictions.jsonl \
  --checkpoint-root runs/m7_reranker/capability_sequence_eval \
  --output-root runs/m7_reranker/predictions/capability_sequence_eval \
  --top-k 100 \
  --device cuda \
  --swanlab-project SkillRQ-M7 \
  --swanlab-run-name m7-capability-sequence-test-predict
```

M7 offline 评估：

```bash
python3 -m skillrq m7 evaluate \
  --prediction-path runs/m7_reranker/predictions/capability/reranked_predictions.jsonl \
  --output-path reports/tables/m7_tool_reranking.json \
  --top-k 5,10,20,50,100 \
  --set-metric-name tool_set_recall
```

M7 sequence-test 评估：

```bash
python3 -m skillrq m7 evaluate \
  --prediction-path runs/m7_reranker/predictions/capability_sequence_eval/reranked_predictions.jsonl \
  --output-path reports/tables/m7_sequence_test_reranking.json \
  --top-k 5,10,20,50,100 \
  --set-metric-name tool_set_recall
```

Code-Aware M7 评估：

```bash
python3 -m skillrq m7 evaluate \
  --prediction-path runs/m7_code_aware_reranker/predictions/capability_sequence_eval_m4_prior/reranked_predictions.jsonl \
  --output-path reports/tables/m7_code_aware_sequence_test_reranking.json \
  --top-k 5,10,20,50,100 \
  --set-metric-name tool_set_recall
```

### 4.4 Code-Path-Guided Prompt Construction

使用 Code-Aware M7 reranking 结果和对应 M5 code plan 构建 LLM agent planning prompt：

```bash
python3 -m skillrq prompt build \
  --prediction-path runs/m7_code_aware_reranker/predictions/capability_sequence_eval_m4_prior/reranked_predictions.jsonl \
  --m5-prediction-path runs/m5_code_path_planner/predictions/capability_sequence_eval_m4_prior/predictions.jsonl \
  --output-root runs/prompt_construction/capability_sequence_eval_m4_prior \
  --top-tools-per-step 3 \
  --max-steps 6
```

输出：

```text
runs/prompt_construction/capability_sequence_eval_m4_prior/
├── prompt_records.jsonl
├── prompts.md
└── prompt_summary.json
```

### 4.5 Mock Tool-Use Simulation

先用 mock simulator 跑通 prompt-grounded tool-call plan 生成和评估链路：

```bash
python3 -m skillrq agent-sim mock \
  --prompt-record-path runs/prompt_construction/capability_sequence_eval_m4_prior/prompt_records.jsonl \
  --output-root runs/agent_sim/mock/capability_sequence_eval_m4_prior \
  --max-calls 6 \
  --tools-per-step 1

python3 -m skillrq agent-sim evaluate \
  --plan-path runs/agent_sim/mock/capability_sequence_eval_m4_prior/tool_call_plans.jsonl \
  --output-path reports/tables/agent_sim_mock_capability_sequence_eval_m4_prior.json \
  --top-k 1,3,5,10
```

### 4.6 vLLM Tool-Use Simulation

真实 LLM tool-call plan 推理使用 vLLM offline batch inference。示例：

```bash
python3 -m skillrq agent-sim vllm \
  --prompt-record-path runs/prompt_construction/capability_sequence_eval_m4_prior/prompt_records.jsonl \
  --output-root runs/agent_sim/vllm/capability_sequence_eval_m4_prior \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --tensor-parallel-size 1 \
  --dtype auto \
  --gpu-memory-utilization 0.90 \
  --temperature 0.0 \
  --top-p 1.0 \
  --max-tokens 512 \
  --batch-size 32
```

如果模型需要自定义代码，可加：

```bash
--trust-remote-code
```

评估 vLLM 生成的 tool-call plans：

```bash
python3 -m skillrq agent-sim evaluate \
  --plan-path runs/agent_sim/vllm/capability_sequence_eval_m4_prior/tool_call_plans.jsonl \
  --output-path reports/tables/agent_sim_vllm_capability_sequence_eval_m4_prior.json \
  --top-k 1,3,5,10
```

vLLM 输出：

```text
runs/agent_sim/vllm/capability_sequence_eval_m4_prior/
├── tool_call_plans.jsonl
├── raw_generations.jsonl
└── vllm_summary.json
```

`raw_generations.jsonl` 会保留原始 LLM 输出，方便分析 JSON 解析失败、工具幻觉和 prompt grounding 问题。

### 4.7 M7 Joint 消融训练

四组消融已经写入脚本：

```bash
cd /hy-tmp
source .venv/bin/activate
bash scripts/run_m7_joint_ablations.sh
```

脚本会依次输出到：

```text
runs/m7_joint_reranker/capability/joint_base
runs/m7_joint_reranker/capability/shared_encoder
runs/m7_joint_reranker/capability/soft_code_distribution
runs/m7_joint_reranker/capability/shared_encoder_soft_code
```

可以覆盖训练参数：

```bash
EPOCHS=20 BATCH_SIZE=4096 DEVICE=cuda bash scripts/run_m7_joint_ablations.sh
```

运行 sequence-aware 评估视图下的四组 joint 消融：

```bash
DATA_ROOT=data/processed/m7_sequence_eval/capability \
OUTPUT_ROOT_BASE=runs/m7_joint_reranker/capability_sequence_eval \
EPOCHS=10 BATCH_SIZE=2048 DEVICE=cuda \
bash scripts/run_m7_joint_ablations.sh
```

首次在新实例上建议小 batch smoke test：

```bash
DEBUG_CUDA=1 EPOCHS=1 BATCH_SIZE=16 bash scripts/run_m7_joint_ablations.sh
```

如果 smoke test 通过，再跑正式训练。

---

## 5. M7 Joint 推理与评估

四个 joint checkpoint 训练完成后，分别执行：

```bash
python3 -m skillrq m7 joint-predict \
  --target capability \
  --prediction-path runs/m5_residual_selector/predictions/capability/predictions.jsonl \
  --checkpoint-root runs/m7_joint_reranker/capability/joint_base \
  --output-root runs/m7_joint_reranker/predictions/capability/joint_base \
  --top-k 100 \
  --device cuda \
  --swanlab-project SkillRQ-M7 \
  --swanlab-run-name m7-joint-base-predict

python3 -m skillrq m7 evaluate \
  --prediction-path runs/m7_joint_reranker/predictions/capability/joint_base/reranked_predictions.jsonl \
  --output-path reports/tables/m7_joint_base.json \
  --top-k 5,10,20,50,100 \
  --set-metric-name tool_set_recall

python3 -m skillrq m7 joint-predict \
  --target capability \
  --prediction-path runs/m5_residual_selector/predictions/capability/predictions.jsonl \
  --checkpoint-root runs/m7_joint_reranker/capability/shared_encoder \
  --output-root runs/m7_joint_reranker/predictions/capability/shared_encoder \
  --top-k 100 \
  --device cuda \
  --swanlab-project SkillRQ-M7 \
  --swanlab-run-name m7-shared-encoder-predict

python3 -m skillrq m7 evaluate \
  --prediction-path runs/m7_joint_reranker/predictions/capability/shared_encoder/reranked_predictions.jsonl \
  --output-path reports/tables/m7_shared_encoder.json \
  --top-k 5,10,20,50,100 \
  --set-metric-name tool_set_recall

python3 -m skillrq m7 joint-predict \
  --target capability \
  --prediction-path runs/m5_residual_selector/predictions/capability/predictions.jsonl \
  --checkpoint-root runs/m7_joint_reranker/capability/soft_code_distribution \
  --output-root runs/m7_joint_reranker/predictions/capability/soft_code_distribution \
  --top-k 100 \
  --device cuda \
  --swanlab-project SkillRQ-M7 \
  --swanlab-run-name m7-soft-code-distribution-predict

python3 -m skillrq m7 evaluate \
  --prediction-path runs/m7_joint_reranker/predictions/capability/soft_code_distribution/reranked_predictions.jsonl \
  --output-path reports/tables/m7_soft_code_distribution.json \
  --top-k 5,10,20,50,100 \
  --set-metric-name tool_set_recall

python3 -m skillrq m7 joint-predict \
  --target capability \
  --prediction-path runs/m5_residual_selector/predictions/capability/predictions.jsonl \
  --checkpoint-root runs/m7_joint_reranker/capability/shared_encoder_soft_code \
  --output-root runs/m7_joint_reranker/predictions/capability/shared_encoder_soft_code \
  --top-k 100 \
  --device cuda \
  --swanlab-project SkillRQ-M7 \
  --swanlab-run-name m7-shared-encoder-soft-code-predict

python3 -m skillrq m7 evaluate \
  --prediction-path runs/m7_joint_reranker/predictions/capability/shared_encoder_soft_code/reranked_predictions.jsonl \
  --output-path reports/tables/m7_shared_encoder_soft_code.json \
  --top-k 5,10,20,50,100 \
  --set-metric-name tool_set_recall
```

将 `joint_base` 替换为：

```text
shared_encoder
soft_code_distribution
shared_encoder_soft_code
```

即可评估另外三组。

---

## 6. 故障排查

### 6.1 CUDA device-side assert / F.embedding 越界

先确认服务器代码已同步最新补丁：

```bash
python3 - <<'PY'
import inspect
from skillrq.m7.joint_model import build_joint_reranker_model
src = inspect.getsource(build_joint_reranker_model)
print("has MeanTextEncoder =", "MeanTextEncoder" in src)
print("has clamp =", "token_ids.clamp" in src)
print("has EmbeddingBag =", "EmbeddingBag" in src)
PY
```

如果 `has EmbeddingBag = True`，说明还是旧版代码，需要重新上传 `skillrq/m7/joint_model.py`。

使用同步 CUDA 定位：

```bash
DEBUG_CUDA=1 EPOCHS=1 BATCH_SIZE=16 bash scripts/run_m7_joint_ablations.sh
```

### 6.2 SwanLab 不可用

临时关闭：

```bash
python3 -m skillrq m7 joint-train \
  --target capability \
  --epochs 1 \
  --batch-size 16 \
  --disable-swanlab \
  --device cuda
```

### 6.3 显存不足

优先降低：

```bash
BATCH_SIZE=512 bash scripts/run_m7_joint_ablations.sh
```

必要时降低：

```bash
EMBEDDING_DIM=256 HIDDEN_DIM=512 BATCH_SIZE=512 bash scripts/run_m7_joint_ablations.sh
```

### 6.4 检查数据是否存在

```bash
ls -lh data/processed/m7/capability/rerank_examples.jsonl
ls -lh data/processed/m4/capability/candidates.jsonl
```

---

## 7. 下载结果后运行诊断实验

当服务器完成 M4 / M5 / M7 / M7 joint 的训练、推理、评估后，将服务器的 `runs/` 和 `reports/` 下载回本地：

```bash
cd /Users/sihan/code/SkillRQ

rsync -av \
  -e "ssh -p ${SERVER_PORT}" \
  "${SERVER_HOST}:${SERVER_DIR}/runs/" \
  runs/

rsync -av \
  -e "ssh -p ${SERVER_PORT}" \
  "${SERVER_HOST}:${SERVER_DIR}/reports/" \
  reports/
```

然后在本地运行诊断：

```bash
python3 -m skillrq diagnostics run \
  --target capability \
  --top-k 5,10,20,50,100
```

输出目录：

```text
reports/diagnostics/capability/
```

核心报告：

```text
candidate_pool_upper_bounds.json       # observed vs oracle Recall/Completeness
candidate_pool_failure_cases.jsonl     # candidate pool 没覆盖 gold 的失败样例
codebook_diagnostics.json              # gold path 分散度、path 混杂度、query-code 弱对齐
multi_positive_diagnostics.json        # 多 gold / 多 path / sequence 分布
negative_sampling_diagnostics.json     # M7 hard negative 难度与 feature 差异
sequence_chain_diagnostics.json        # sequence_ids 是否能进入评估链路
diagnostics_summary.json               # 总入口
```

---

## 8. 建议执行顺序总结

新实例只跑 M7 joint 消融：

```text
1. 上传代码 + data/processed/m7/capability
2. 建 uv 环境 + 安装 torch / swanlab
3. 运行补丁检查，确认 MeanTextEncoder / clamp 生效
4. DEBUG_CUDA=1 EPOCHS=1 BATCH_SIZE=16 bash scripts/run_m7_joint_ablations.sh
5. EPOCHS=10 BATCH_SIZE=2048 bash scripts/run_m7_joint_ablations.sh
```

完整 pipeline：

```text
1. 上传代码 + data/processed
2. 训练 soft multi-path M4
3. M4 predict/evaluate
4. 准备并训练 M5 code-plan
5. M5 code-plan m4-prior predict/evaluate
6. 训练 M7 standard / Code-Aware reranker
7. 对 M5 code-plan predictions 做 M7 reranking/evaluate
8. 构建 code-path-guided prompt
9. 运行 mock agent simulation/evaluate
10. 可选：运行 vLLM agent simulation/evaluate
11. 可选：训练 M7 joint ablations
12. 下载 runs/reports 并运行 diagnostics
```
