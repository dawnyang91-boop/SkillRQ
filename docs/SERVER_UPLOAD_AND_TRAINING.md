# SkillRQ 服务器上传与训练指南

本文档用于把本地 `/Users/sihan/code/SkillRQ` 迁移到新的 GPU 服务器，并在服务器上执行 M4 / M5 / M7 训练、推理与评估。

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
export SERVER_PORT=47046
export SERVER_HOST=root@connect.nmb2.seetacloud.com
export SERVER_DIR=/root/autodl-tmp
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

---

## 3. 服务器环境初始化

以下命令在服务器执行。

```bash
cd /root/autodl-tmp/SkillRQ
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

### 4.1 M4 Query-to-Code 训练

```bash
cd /hy-tmp/SkillRQ
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

M4 评估：

```bash
python3 -m skillrq m4 evaluate \
  --prediction-path runs/m4_query_to_code/predictions/capability/predictions.jsonl \
  --output-path reports/tables/m4_capabilityrq_capability_metrics.json \
  --top-k 1,5,10,20,50,100 \
  --set-metric-name tool_set_recall
```

### 4.2 M5 Residual Selector 训练

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

M5 评估：

```bash
python3 -m skillrq m5 evaluate \
  --prediction-path runs/m5_residual_selector/predictions/capability/predictions.jsonl \
  --output-path reports/tables/m5_coverage_supervision_capability.json \
  --top-k 5,10,20,50,100 \
  --set-metric-name tool_set_recall
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

M7 offline 评估：

```bash
python3 -m skillrq m7 evaluate \
  --prediction-path runs/m7_reranker/predictions/capability/reranked_predictions.jsonl \
  --output-path reports/tables/m7_tool_reranking.json \
  --top-k 5,10,20,50,100 \
  --set-metric-name tool_set_recall
```

### 4.4 M7 Joint 消融训练

四组消融已经写入脚本：

```bash
cd /hy-tmp/SkillRQ
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

## 7. 建议执行顺序总结

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
2. 训练 M4
3. M4 predict/evaluate
4. 训练 M5
5. M5 predict/evaluate
6. 训练 M7 offline
7. 训练 M7 joint ablations
8. 对 M5 predictions 做 M7 / M7 joint reranking
9. evaluate 汇总 metrics
```
