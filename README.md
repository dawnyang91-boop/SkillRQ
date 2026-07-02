# SkillRQ

SkillRQ / CapabilityRQ 是一个面向 LLM Agent capability selection 的研究型工程项目。当前研究任务已经从单一 Skill Recommendation 扩展为统一的 Agent Capability Recommendation：Agent 可调用的能力单元既可以是高层封装的 skills，也可以是更细粒度的 tools / APIs / functions。

后续实验以 Tool/API Recommendation 为主实验，以 Skill Recommendation 为副实验。核心目标是用可解释的残差语义量化替代文本级 query span decomposition，让完整用户 query 在 capability semantic space 中生成多个 code paths，再结合 hypergraph 补齐隐含协作 tools / skills，最终输出 role-aware capability recommendation。

本目录当前包含项目执行计划与建议文件架构，后续实现应优先遵循：

- [执行计划](docs/EXECUTION_PLAN.md)
- [项目文件架构](docs/PROJECT_STRUCTURE.md)

## 数据位置

历史 skill 原始数据位于：

```text
/Users/sihan/code/skill-rec/data/raw/
```

新增 tool/API 原始数据位于：

```text
/Users/sihan/code/SkillRQ/data/raw/
```

已确认的主要数据源：

- `ToolBench/data`: 主实验数据集，用于 tool/API recommendation、multi-tool coverage、execution-stage modeling 和 hypergraph completion。
- `DAMO-ConvAI/api-bank`: 补充验证集，用于多轮 tool-use、API 调用链和较小规模工具库泛化评估。
- `skillret/`: `train`、`test` 下有 `skills.jsonl`、`queries.jsonl`、`qrels.jsonl`，适合作为第一阶段的 supervised retrieval 数据。
- `skillrouter/eval_core/`: 包含 `tasks.jsonl`、`relevance.json`、`easy/`、`hard/` 分片，适合补充泛化评测与 hard negative。
- `sra_bench/bench/`: 包含 `corpus/corpus.json` 与多个 `instances/*.json`，适合多技能覆盖、implicit skill 和 agent-level 评测。
- `skillsbench`: 仅作为最终 Agent 端到端评测候选数据集，当前阶段不下载、不作为训练依赖。

## 推荐推进顺序

1. Phase 1: 用 ToolBench 训练 CapabilityRQ 主体，覆盖 capability encoder、codebook、query-to-code、candidate retrieval、multi-tool coverage 和 hypergraph expansion。
2. Phase 2: 用 API-Bank 验证多轮 tool-use、API dependency、tool sequence ordering 和跨数据分布泛化。
3. Phase 3: 用 SkillRet / SkillRouter 做 skill-level 副实验，验证高层 skill library 上的泛化能力，默认关闭 hypergraph。
4. Phase 4: 用 SkillsBench 做最终 Agent 端到端验证，比较 flat skills、role support、code support 和 hypergraph expansion 对 planner 的影响。

## 当前可执行指令

以下命令均在项目根目录执行：

```bash
cd /Users/sihan/code/SkillRQ
```

### 1. 基础检查

```bash
python3 -m skillrq --help
python3 -m skillrq config show
python3 -m skillrq m2 run --help
```

### 2. 构建 SkillRet skill-level processed data

```bash
python3 -m skillrq data build --dataset skillret
```

该命令从 `/Users/sihan/code/skill-rec/data/raw/skillret` 读取原始数据，并写入：

```text
data/processed/
```

主要产物：

```text
data/processed/skills.jsonl
data/processed/queries.jsonl
data/processed/qrels.jsonl
data/processed/task_skill_sets.jsonl
data/processed/roles.jsonl
data/processed/stats.json
```

### 3. 构建 ToolBench / API-Bank capability-level processed data

API-Bank 全量构建：

```bash
python3 -m skillrq capability build --dataset api_bank
```

ToolBench smoke test：

```bash
python3 -m skillrq capability build --dataset toolbench --skip-answer-trees --limit-tools 100 --limit-queries 50
```

ToolBench + API-Bank smoke test：

```bash
python3 -m skillrq capability build --dataset all --skip-answer-trees --limit-tools 100 --limit-queries 50
```

ToolBench + API-Bank 全量构建：

```bash
python3 -m skillrq capability build --dataset all
```

`capability build` 会从 `/Users/sihan/code/SkillRQ/data/raw` 读取 ToolBench / API-Bank，并写入：

```text
data/processed/capability/
```

主要产物：

```text
data/processed/capability/capabilities.jsonl
data/processed/capability/capability_queries.jsonl
data/processed/capability/capability_qrels.jsonl
data/processed/capability/capability_sequences.jsonl
data/processed/capability/capability_stats.json
```

当前已完成的全量构建规模：

```text
capabilities = 64,645
queries      = 329,625
qrels        = 813,490
sequences    = 447,066
```

### 4. 生成 capability processed data 报告

```bash
python3 scripts/write_capability_data_report.py
```

输出：

```text
docs/CAPABILITY_DATA_REPORT.md
```

### 5. 运行 M2 retrieval baselines

默认运行四个数据集和两类 baseline：

```bash
python3 -m skillrq m2 run
```

默认配置：

```text
datasets       = toolbench, api_bank, skillret, skillrouter
methods        = bm25, dense
top_k          = 1,5,10,20
max_queries    = 300
max_candidates = 10000
```

默认命令会生成：

```text
runs/m2_baseline_retrieval/
reports/data_stats/
```

其中每个 dataset/method 都会包含：

```text
metrics.json
predictions.jsonl
run_config.json
```

只跑 ToolBench 与 API-Bank：

```bash
python3 -m skillrq m2 run --datasets toolbench api_bank
```

只跑 BM25：

```bash
python3 -m skillrq m2 run --methods bm25
```

小规模 smoke test：

```bash
python3 -m skillrq m2 run \
  --datasets api_bank \
  --methods bm25 dense \
  --max-queries 20 \
  --max-candidates 200 \
  --top-k 1,5 \
  --run-root /private/tmp/skillrq_m2_smoke
```

全量查询/候选运行：

```bash
python3 -m skillrq m2 run --max-queries 0 --max-candidates 0
```

注意：全量 ToolBench / SkillRouter 会明显更慢。默认配置会截取 queries/candidates，但 gold candidates 始终强制保留，不会因为候选截断丢失 gold label。

### 6. 构建 M3 CapabilityRQ semantic codebook

默认对 ToolBench、API-Bank、SkillRet、SkillRouter 生成 semantic code assignments：

```bash
python3 -m skillrq m3 build
```

输出：

```text
data/processed/capability/code_assignments.jsonl
data/processed/capability/code_quality.json
data/processed/skill/code_assignments.jsonl
data/processed/skill/code_quality.json
reports/code_cards/
```

小规模 smoke test：

```bash
python3 -m skillrq m3 build \
  --datasets toolbench api_bank skillret \
  --limit-per-dataset 20
```

### 7. 准备并训练 M4 Query-to-Code 模型

M4 使用 PyTorch 训练 query-to-code 模型。云服务器建议先安装训练依赖：

```bash
pip install -e ".[train]"
```

准备 ToolBench + API-Bank capability-level 训练数据：

```bash
python3 -m skillrq m4 prepare \
  --target capability \
  --datasets toolbench api_bank
```

准备 SkillRet skill-level 训练数据：

```bash
python3 -m skillrq m4 prepare \
  --target skill \
  --datasets skillret
```

当前已生成的 M4 训练数据规模：

```text
capability: candidates=64,645 queries=329,207 train_pairs=813,445
skill:      candidates=16,783 queries=68,256  train_pairs=135,537
```

从 capability train split 中划出带 tool sequence 的 query，构建 sequence-aware dev/test 评估视图：

```bash
python3 -m skillrq m4 sequence-split \
  --target capability \
  --sequence-dev-size 2000 \
  --sequence-test-size 5000 \
  --seed 13
```

输出：

```text
data/processed/m4_sequence_eval/capability/
```

当前已生成的 sequence-aware M4 视图规模：

```text
train=321,107 sequence_dev=2,000 sequence_test=5,000 test=1,100
```

训练 CapabilityRQ query-to-code 主模型：

```bash
python3 -m skillrq m4 train \
  --target capability \
  --epochs 20 \
  --batch-size 2048 \
  --learning-rate 3e-4 \
  --embedding-dim 512 \
  --hidden-dim 1024 \
  --device cuda
```

训练 sequence-aware 评估视图下的 CapabilityRQ query-to-code 模型：

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
  --device cuda
```

显式启用新版 soft multi-path M4。该分支将 query 作为 multi-positive 样本训练，学习 code path distribution，并加入 hierarchical code prediction 与 query-code contrastive alignment：

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

训练 sequence-aware 评估视图下的 soft multi-path M4：

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

训练 SkillRet 副实验 query-to-code 模型：

```bash
python3 -m skillrq m4 train \
  --target skill \
  --epochs 20 \
  --batch-size 1024 \
  --learning-rate 3e-4 \
  --embedding-dim 512 \
  --hidden-dim 1024 \
  --device cuda
```

用训练好的 CapabilityRQ 模型预测 code paths 并检索 candidates：

```bash
python3 -m skillrq m4 predict \
  --target capability \
  --checkpoint-root runs/m4_query_to_code/capabilityrq/capability \
  --output-root runs/m4_query_to_code/predictions/capability \
  --top-n-paths 16 \
  --candidate-budget 100 \
  --split test \
  --device cuda
```

评估 CapabilityRQ 预测结果：

```bash
python3 -m skillrq m4 evaluate \
  --prediction-path runs/m4_query_to_code/predictions/capability/predictions.jsonl \
  --output-path reports/tables/m4_capabilityrq_capability_metrics.json \
  --top-k 1,5,10,20,50,100 \
  --set-metric-name tool_set_recall
```

用 soft multi-path M4 预测 code path distribution 并检索 candidates：

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

soft multi-path 预测输出中的每条 `predicted_code_paths` 会包含：

```text
semantic_id
codes
probability
hierarchy_probability
contrastive_probability
reason
verbalization
code_explanation
```

训练 RQ-KMeans code retrieval 对照方法：

```bash
python3 -m skillrq m4 rq-kmeans \
  --target capability \
  --num-levels 4 \
  --codebook-size 512 \
  --iterations 50 \
  --feature-dim 4096 \
  --device cuda
```

训练 Ordinary RQ-VAE code retrieval 对照方法：

```bash
python3 -m skillrq m4 rq-vae \
  --target capability \
  --epochs 50 \
  --batch-size 4096 \
  --learning-rate 1e-4 \
  --feature-dim 4096 \
  --latent-dim 512 \
  --num-levels 4 \
  --codebook-size 512 \
  --device cuda
```

M4 主要输出：

```text
data/processed/m4/capability/
data/processed/m4/skill/
runs/m4_query_to_code/
runs/m4_query_to_code/soft_multipath/
reports/tables/m4_capabilityrq_capability_metrics.json
```

soft multi-path M4 的 SwanLab 会额外记录 `train/hierarchy_loss`、`train/contrastive_loss`、`train/path_bce_loss`、`dev/path_recall@16`、`dev/path_top1_accuracy`，用于观察多路径监督、层级预测和 query-code 对齐是否同时收敛。

### 8. 准备并训练 M5 Residual Multi-Code Path Selector

M5 在 M4 code paths 基础上训练 residual selector。它不再一次性给出 flat code list，而是逐步预测多个 code paths，并用 coverage supervision 约束每一步尽量覆盖新的 gold tools / skills。

当前 M5 有两种显式分支：

```text
--model-kind coverage   # 旧版 residual candidate coverage baseline
--model-kind code-plan  # 新版 residual code path planning 主分支
```

准备 capability-level coverage supervision 数据：

```bash
python3 -m skillrq m5 prepare \
  --target capability \
  --max-steps 6
```

基于 sequence-aware M4 视图准备 capability-level coverage supervision 数据：

```bash
python3 -m skillrq m5 prepare \
  --target capability \
  --m4-data-root data/processed/m4_sequence_eval/capability \
  --output-root data/processed/m5_sequence_eval/capability \
  --max-steps 6
```

准备 skill-level coverage supervision 数据：

```bash
python3 -m skillrq m5 prepare \
  --target skill \
  --max-steps 6
```

当前已生成的 M5 训练数据规模：

```text
capability: queries=329,207 residual_examples=798,736 avg_steps_per_query=2.4262
skill:      queries=68,256  residual_examples=135,185 avg_steps_per_query=1.9806
```

当前已生成的 sequence-aware M5 视图规模：

```text
train=778,957 sequence_dev=4,966 sequence_test=12,236 test=2,577
```

基于 soft multi-path M4 predictions 准备 residual code path planning 数据：

```bash
python3 -m skillrq m5 prepare \
  --target capability \
  --model-kind code-plan \
  --m4-data-root data/processed/m4_sequence_eval/capability \
  --m4-prediction-path runs/m4_query_to_code/predictions/soft_multipath/capability_sequence_eval/predictions.jsonl \
  --output-root data/processed/m5_code_plan/capability_sequence_eval \
  --max-steps 6
```

如果暂时没有 M4 predictions，可以不传 `--m4-prediction-path`，脚本会用 gold code paths 构造 oracle planning 数据，适合先做训练 smoke test。

训练 capability-level residual selector：

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

训练 sequence-aware 评估视图下的 residual selector：

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

训练 skill-level residual selector：

```bash
python3 -m skillrq m5 train \
  --target skill \
  --epochs 20 \
  --batch-size 1024 \
  --learning-rate 3e-4 \
  --embedding-dim 512 \
  --hidden-dim 1024 \
  --coverage-weight 1.0 \
  --device cuda \
  --swanlab-project SkillRQ-M5 \
  --swanlab-run-name m5-skill-coverage
```

训练 residual code path planner：

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

用 M5 模型预测 residual code paths 并检索 candidates：

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

使用 residual code path planner 输出 `code_plan`：

```bash
python3 -m skillrq m5 predict \
  --target capability \
  --model-kind code-plan \
  --m4-data-root data/processed/m4_sequence_eval/capability \
  --m4-prediction-path runs/m4_query_to_code/predictions/soft_multipath/capability_sequence_eval/predictions.jsonl \
  --checkpoint-root runs/m5_code_path_planner/capability_sequence_eval \
  --output-root runs/m5_code_path_planner/predictions/capability_sequence_eval \
  --max-steps 6 \
  --top-n-paths 16 \
  --candidates-per-step 20 \
  --stop-threshold 0.55 \
  --split sequence_test \
  --device cuda \
  --swanlab-project SkillRQ-M5 \
  --swanlab-run-name m5-code-path-plan-sequence-test
```

`code-plan` 推理输出会包含：

```text
code_plan[].code_path
code_plan[].role
code_plan[].purpose
code_plan[].expected_coverage_gain
code_plan[].stop_probability
```

同时保留 `residual_code_paths` 字段，方便后续 M7 沿用现有 prediction 输入格式。

评估 M5 coverage prediction：

```bash
python3 -m skillrq m5 evaluate \
  --prediction-path runs/m5_residual_selector/predictions/capability/predictions.jsonl \
  --output-path reports/tables/m5_coverage_supervision_capability.json \
  --top-k 5,10,20,50,100 \
  --set-metric-name tool_set_recall
```

M5 evaluate 偏低时，先运行 phase 0 诊断脚本确认断点位置：

```bash
python3 scripts/diagnose_m5_nested_candidates.py \
  --prediction-path runs/m5_code_path_planner/predictions/capability_sequence_eval/predictions.jsonl \
  --output-path reports/tables/m5_nested_candidates_diagnostic.json

python3 scripts/diagnose_m5_codepath_to_candidate.py \
  --prediction-path runs/m5_code_path_planner/predictions/capability_sequence_eval/predictions.jsonl \
  --m4-data-root data/processed/m4_sequence_eval/capability \
  --output-path reports/tables/m5_codepath_to_candidate_diagnostic.json

python3 scripts/diagnose_m4_m5_candidate_reuse.py \
  --m4-prediction-path runs/m4_query_to_code/predictions/soft_multipath/capability_sequence_eval/predictions.jsonl \
  --m5-prediction-path runs/m5_code_path_planner/predictions/capability_sequence_eval/predictions.jsonl \
  --output-path reports/tables/m4_m5_candidate_reuse_diagnostic.json
```

M5 主要输出：

```text
data/processed/m5/capability/
data/processed/m5/skill/
data/processed/m5_code_plan/
runs/m5_residual_selector/
runs/m5_code_path_planner/
reports/tables/m5_coverage_supervision_capability.json
```

SwanLab 会记录每个 epoch 的 `train/loss`、`train/code_loss`、`train/coverage_loss`、`dev/loss`、`dev/code_loss`、`dev/coverage_loss`、`dev/l1_accuracy`、`dev/l2_accuracy`、`dev/l3_accuracy`、`dev/l4_accuracy`、`dev/path_exact_match`，以及推理阶段的 `predict/queries`、`predict/avg_steps`。

`code-plan` 分支额外记录 `train/role_loss`、`train/stop_loss`、`dev/role_accuracy`、`dev/stop_accuracy`，用于观察 role planning 和 learned stopping 是否有效。

Phase 2 之后，`m5 predict --model-kind code-plan` 会复用 M4 soft multi-path 输出中的 `retrieved_capabilities`。推理 summary 与 SwanLab 会额外记录 `m4_candidate_reuse_rate`、`m4_hit_rate`、`m5_hit_rate`、`m4_hit_m5_miss_rate` 和 `m4_miss_m5_hit_rate`，用于判断 M5 是否仍在丢弃 M4 已召回的 gold candidates。

为提高 M5 predict 速度，code-plan 分支会在启动时构建 candidate retrieval index：`exact_path -> candidates`、`L1/L2/L3 prefix -> candidates`、`L1/L2 prefix -> candidates` 和 `candidate_id -> candidate`。推理时 `_retrieve_for_path()` 只访问相关 bucket 并合并 M4 candidates，不再每个 step 遍历全量 candidate 库。

如果要启用 Phase 1 的 exact-first bucket retrieval：

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
  --device cuda
```

若要做 Phase 1 only 消融，可额外加 `--disable-m4-candidate-prior`。

### 9. 准备并训练 M7 Role-Aware and Sequence-Aware Reranker

M7 跳过 M6 hypergraph optional branch，直接对 M4/M5 产生的候选池做 role-aware 与 sequence-aware reranking。`hypergraph_support_score` 特征位会保留，但当前默认是 `0.0`。

准备 capability-level reranker 数据：

```bash
python3 -m skillrq m7 prepare \
  --target capability \
  --negatives-per-positive 2
```

基于 sequence-aware M4 视图准备 capability-level reranker 数据：

```bash
python3 -m skillrq m7 prepare \
  --target capability \
  --m4-data-root data/processed/m4_sequence_eval/capability \
  --output-root data/processed/m7_sequence_eval/capability \
  --negatives-per-positive 2
```

准备 skill-level reranker 数据：

```bash
python3 -m skillrq m7 prepare \
  --target skill \
  --negatives-per-positive 2
```

当前已生成的 M7 训练数据规模：

```text
capability: queries=329,207 examples=2,440,301 positives=813,445 negatives=1,626,856 queries_with_sequence=126,332
skill:      queries=68,256  examples=406,611   positives=135,537 negatives=271,074   queries_with_sequence=0
```

当前已生成的 sequence-aware M7 视图规模：

```text
train=2,379,907 sequence_dev=15,174 sequence_test=37,333 test=7,887
```

训练 capability-level reranker：

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

训练 sequence-aware 评估视图下的 reranker：

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

训练 Code-Aware M7 reranker。该分支会使用结构化输入块 `[User Query]`、`[Predicted Code Path]`、`[Code Path Explanation]`、`[Candidate Tool/API]`、`[Candidate Schema]`、`[Candidate Native Code Path]`、`[Role Requirement]` 和 `[Coverage State]`，并额外学习 code consistency、schema compatibility、coverage gain 和 prompt usefulness：

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

训练 skill-level reranker：

```bash
python3 -m skillrq m7 train \
  --target skill \
  --epochs 10 \
  --batch-size 1024 \
  --learning-rate 3e-4 \
  --embedding-dim 512 \
  --hidden-dim 1024 \
  --role-weight 0.2 \
  --stage-weight 0.2 \
  --order-weight 0.0 \
  --device cuda \
  --swanlab-project SkillRQ-M7 \
  --swanlab-run-name m7-skill-reranker
```

M7 joint 消融训练。默认不启用 shared encoder，也不启用 soft code distribution：

```bash
python3 -m skillrq m7 joint-train \
  --target capability \
  --epochs 10 \
  --batch-size 2048 \
  --learning-rate 3e-4 \
  --embedding-dim 512 \
  --hidden-dim 1024 \
  --code-embedding-dim 128 \
  --code-weight 1.0 \
  --role-weight 0.2 \
  --stage-weight 0.2 \
  --order-weight 0.2 \
  --device cuda \
  --swanlab-project SkillRQ-M7 \
  --swanlab-run-name m7-joint-base
```

四组 joint 消融也可以用脚本一次跑完；如果要使用 sequence-aware 视图，可指定 `DATA_ROOT` 与 `OUTPUT_ROOT_BASE`：

```bash
DATA_ROOT=data/processed/m7_sequence_eval/capability \
OUTPUT_ROOT_BASE=runs/m7_joint_reranker/capability_sequence_eval \
bash scripts/run_m7_joint_ablations.sh
```

启用 shared query encoder + code encoder 分支：

```bash
python3 -m skillrq m7 joint-train \
  --target capability \
  --epochs 10 \
  --batch-size 2048 \
  --learning-rate 3e-4 \
  --embedding-dim 512 \
  --hidden-dim 1024 \
  --enable-shared-encoder \
  --device cuda \
  --swanlab-project SkillRQ-M7 \
  --swanlab-run-name m7-joint-shared-encoder
```

启用 soft code distribution 分支：

```bash
python3 -m skillrq m7 joint-train \
  --target capability \
  --epochs 10 \
  --batch-size 2048 \
  --learning-rate 3e-4 \
  --embedding-dim 512 \
  --hidden-dim 1024 \
  --enable-soft-code-distribution \
  --soft-code-weight 0.1 \
  --device cuda \
  --swanlab-project SkillRQ-M7 \
  --swanlab-run-name m7-joint-soft-code
```

同时启用两个分支：

```bash
python3 -m skillrq m7 joint-train \
  --target capability \
  --epochs 10 \
  --batch-size 2048 \
  --learning-rate 3e-4 \
  --embedding-dim 512 \
  --hidden-dim 1024 \
  --enable-shared-encoder \
  --enable-soft-code-distribution \
  --soft-code-weight 0.1 \
  --device cuda \
  --swanlab-project SkillRQ-M7 \
  --swanlab-run-name m7-joint-shared-soft
```

对 M5 capability predictions 进行 reranking：

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

使用 Code-Aware M7 对 Phase 2 M5 predictions 进行 reranking：

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

使用 joint 模型对 M5 capability predictions 进行 reranking：

```bash
python3 -m skillrq m7 joint-predict \
  --target capability \
  --prediction-path runs/m5_residual_selector/predictions/capability/predictions.jsonl \
  --checkpoint-root runs/m7_joint_reranker/capability \
  --output-root runs/m7_joint_reranker/predictions/capability \
  --top-k 100 \
  --device cuda \
  --swanlab-project SkillRQ-M7 \
  --swanlab-run-name m7-joint-capability-predict
```

评估 M7 reranking：

```bash
python3 -m skillrq m7 evaluate \
  --prediction-path runs/m7_reranker/predictions/capability/reranked_predictions.jsonl \
  --output-path reports/tables/m7_tool_reranking.json \
  --top-k 5,10,20,50,100 \
  --set-metric-name tool_set_recall
```

如果要专门评估 sequence-aware 指标，用 `m4_sequence_eval` / `m5_sequence_eval` / `m7_sequence_eval` 数据视图训练，并在推理时指定 `--split sequence_test`：

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
  --device cuda

python3 -m skillrq m7 predict \
  --target capability \
  --m4-data-root data/processed/m4_sequence_eval/capability \
  --prediction-path runs/m5_residual_selector/predictions/capability_sequence_eval/predictions.jsonl \
  --checkpoint-root runs/m7_reranker/capability_sequence_eval \
  --output-root runs/m7_reranker/predictions/capability_sequence_eval \
  --top-k 100 \
  --device cuda

python3 -m skillrq m7 evaluate \
  --prediction-path runs/m7_reranker/predictions/capability_sequence_eval/reranked_predictions.jsonl \
  --output-path reports/tables/m7_sequence_test_reranking.json \
  --top-k 5,10,20,50,100 \
  --set-metric-name tool_set_recall
```

M7 主要输出：

```text
data/processed/m7/capability/
data/processed/m7/skill/
runs/m5_residual_selector/predictions/capability_sequence_eval/
runs/m7_reranker/predictions/capability_sequence_eval/
runs/m7_reranker/
runs/m7_joint_reranker/
reports/tables/m7_tool_reranking.json
```

SwanLab 会记录每个 epoch 的 `train/loss`、`train/relevance_loss`、`train/role_loss`、`train/stage_loss`、`train/order_loss`、`dev/relevance_accuracy`、`dev/role_accuracy`、`dev/stage_accuracy`、`dev/order_mse`，以及推理阶段的 `predict/queries`、`predict/top_k`、`predict/avg_reranked_candidates`。`m7 train --model-kind code-aware` 会额外记录 `train/code_consistency_loss`、`train/schema_loss`、`train/coverage_gain_loss`、`train/prompt_usefulness_loss`、`dev/code_consistency_mse`、`dev/schema_compatibility_mse`、`dev/coverage_gain_mse` 和 `dev/prompt_usefulness_mse`。`m7 joint-train` 还会额外记录 `train/code_loss`、`train/soft_code_loss` 和 `dev/code_path_exact_match`。

### 10. 构建 Code-Path-Guided LLM Prompt

Prompt construction 是正式模块，用于把 retrieval / reranking 结果转化为 LLM agent planner 可直接使用的 code-structured planning support，而不是 flat top-k tool list。

使用 Code-Aware M7 reranking 结果和对应 M5 code plan 构建 prompt：

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

每条 prompt 会包含：

```text
User Query
Capability Plan
Step role / operation
Required capability code path
Candidate tools
Why needed
Planner Instruction
```

### 11. Mock Tool-Use Simulation

先用 mock simulator 跑通 tool call plan 生成与评估链路。该模式不调用真实 LLM，不会访问外部 API，只从 prompt 中给出的 candidate tools 里按 capability step 顺序选择工具，因此可作为 prompt schema / parser / evaluator 的 sanity check。

生成 mock tool-call plans：

```bash
python3 -m skillrq agent-sim mock \
  --prompt-record-path runs/prompt_construction/capability_sequence_eval_m4_prior/prompt_records.jsonl \
  --output-root runs/agent_sim/mock/capability_sequence_eval_m4_prior \
  --max-calls 6 \
  --tools-per-step 1
```

评估 mock plans：

```bash
python3 -m skillrq agent-sim evaluate \
  --plan-path runs/agent_sim/mock/capability_sequence_eval_m4_prior/tool_call_plans.jsonl \
  --output-path reports/tables/agent_sim_mock_capability_sequence_eval_m4_prior.json \
  --top-k 1,3,5,10
```

指标包括：

```text
tool_set_recall@K
completeness@K
first_tool_accuracy
transition_accuracy
invalid_tool_rate
prompt_grounding_rate
avg_tool_calls
```

真实 LLM tool call plan 推理建议作为下一步接入 `skillrq/agent_sim/`，优先支持 vLLM batch inference。推荐输出仍保持同一 JSON schema：`tool_call_plans.jsonl`，这样可以复用当前 evaluator。

使用 vLLM 进行真实 LLM tool-call plan 推理：

```bash
python3 -m skillrq agent-sim vllm \
  --prompt-record-path runs/prompt_construction/capability_sequence_eval_m4_prior/prompt_records.jsonl \
  --output-root runs/agent_sim/vllm/capability_sequence_eval_m4_prior \
  --model Qwen/Qwen2.5-7B-Instruct \
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

### 12. 运行诊断实验与上界分析

下载服务器 `runs/` 和 `reports/` 后，在本地运行：

```bash
python3 -m skillrq diagnostics run \
  --target capability \
  --top-k 5,10,20,50,100
```

输出：

```text
reports/diagnostics/capability/candidate_pool_upper_bounds.json
reports/diagnostics/capability/candidate_pool_failure_cases.jsonl
reports/diagnostics/capability/codebook_diagnostics.json
reports/diagnostics/capability/codebook_query_cases.jsonl
reports/diagnostics/capability/codebook_mixed_path_cases.jsonl
reports/diagnostics/capability/multi_positive_diagnostics.json
reports/diagnostics/capability/negative_sampling_diagnostics.json
reports/diagnostics/capability/hard_negative_cases.jsonl
reports/diagnostics/capability/sequence_chain_diagnostics.json
reports/diagnostics/capability/diagnostics_summary.json
```

这些报告用于判断 first-stage candidate pool 上界、codebook 质量、multi-positive 分布、hard negative 难度与 sequence 评估链路。

### 13. 运行测试

```bash
/opt/homebrew/anaconda3/bin/pytest
```

当前测试结果：

```text
18 passed
```

## 当前已完成阶段

- M0: 项目初始化。
- M1: 数据规范化，包括 SkillRet、ToolBench、API-Bank。
- M2: Capability Retrieval Baselines，包括 BM25、hashing dense、统一 metrics、baseline artifact 与 data stats。
- M3: CapabilityRQ Codebook v1，包括四层 semantic code path、code quality、code assignments 与 code cards。
- M4: Query-to-Code Latent Capability Decomposition，包括 PyTorch query-to-code 训练代码、RQ-KMeans/RQ-VAE 对照入口、预测与评估脚本。
- M5: Residual Multi-Code Path Selector with Coverage Supervision，包括 residual coverage 数据构造、PyTorch selector 训练、逐步推理、coverage metrics 与 SwanLab 记录。
- M7: Role-Aware and Sequence-Aware Reranker，包括 reranker 数据构造、PyTorch relevance/role/stage/order 多任务训练、Code-Aware Reranker、M4/M5 候选池重排、tool order 预测、retrieval/sequence metrics、joint ablation 的 shared encoder / soft code distribution 可选分支与 SwanLab 记录。
- Prompt Construction: code-path-guided LLM prompt construction，将 M5 code plan 与 M7 reranked capabilities 转换为 agent planning prompt。
- Agent Simulation: mock tool-use simulation + evaluation，用于验证 prompt-grounded tool selection、tool order 和 hallucination/grounding。
- Diagnostics: 候选池 oracle 上界、codebook 混杂度、multi-positive 分布、negative sampling 难度与 sequence 评估链路诊断。
- M6: Granularity-Aware Hypergraph Expansion 按当前计划先跳过，作为后续 optional branch。
- LLM query span decomposition + segment retrieval 当前按计划保留为后续可选项。
