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
reports/tables/m4_capabilityrq_capability_metrics.json
```

### 8. 准备并训练 M5 Residual Multi-Code Path Selector

M5 在 M4 code paths 基础上训练 residual selector。它不再一次性给出 flat code list，而是逐步预测多个 code paths，并用 coverage supervision 约束每一步尽量覆盖新的 gold tools / skills。

准备 capability-level coverage supervision 数据：

```bash
python3 -m skillrq m5 prepare \
  --target capability \
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

评估 M5 coverage prediction：

```bash
python3 -m skillrq m5 evaluate \
  --prediction-path runs/m5_residual_selector/predictions/capability/predictions.jsonl \
  --output-path reports/tables/m5_coverage_supervision_capability.json \
  --top-k 5,10,20,50,100 \
  --set-metric-name tool_set_recall
```

M5 主要输出：

```text
data/processed/m5/capability/
data/processed/m5/skill/
runs/m5_residual_selector/
reports/tables/m5_coverage_supervision_capability.json
```

SwanLab 会记录每个 epoch 的 `train/loss`、`train/code_loss`、`train/coverage_loss`、`dev/loss`、`dev/code_loss`、`dev/coverage_loss`、`dev/l1_accuracy`、`dev/l2_accuracy`、`dev/l3_accuracy`、`dev/l4_accuracy`、`dev/path_exact_match`，以及推理阶段的 `predict/queries`、`predict/avg_steps`。

### 9. 准备并训练 M7 Role-Aware and Sequence-Aware Reranker

M7 跳过 M6 hypergraph optional branch，直接对 M4/M5 产生的候选池做 role-aware 与 sequence-aware reranking。`hypergraph_support_score` 特征位会保留，但当前默认是 `0.0`。

准备 capability-level reranker 数据：

```bash
python3 -m skillrq m7 prepare \
  --target capability \
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

M7 主要输出：

```text
data/processed/m7/capability/
data/processed/m7/skill/
runs/m7_reranker/
runs/m7_joint_reranker/
reports/tables/m7_tool_reranking.json
```

SwanLab 会记录每个 epoch 的 `train/loss`、`train/relevance_loss`、`train/role_loss`、`train/stage_loss`、`train/order_loss`、`dev/relevance_accuracy`、`dev/role_accuracy`、`dev/stage_accuracy`、`dev/order_mse`，以及推理阶段的 `predict/queries`、`predict/top_k`、`predict/avg_reranked_candidates`。`m7 joint-train` 还会额外记录 `train/code_loss`、`train/soft_code_loss` 和 `dev/code_path_exact_match`。

### 10. 运行测试

```bash
/opt/homebrew/anaconda3/bin/pytest
```

当前测试结果：

```text
14 passed
```

## 当前已完成阶段

- M0: 项目初始化。
- M1: 数据规范化，包括 SkillRet、ToolBench、API-Bank。
- M2: Capability Retrieval Baselines，包括 BM25、hashing dense、统一 metrics、baseline artifact 与 data stats。
- M3: CapabilityRQ Codebook v1，包括四层 semantic code path、code quality、code assignments 与 code cards。
- M4: Query-to-Code Latent Capability Decomposition，包括 PyTorch query-to-code 训练代码、RQ-KMeans/RQ-VAE 对照入口、预测与评估脚本。
- M5: Residual Multi-Code Path Selector with Coverage Supervision，包括 residual coverage 数据构造、PyTorch selector 训练、逐步推理、coverage metrics 与 SwanLab 记录。
- M7: Role-Aware and Sequence-Aware Reranker，包括 reranker 数据构造、PyTorch relevance/role/stage/order 多任务训练、M4/M5 候选池重排、tool order 预测、retrieval/sequence metrics、joint ablation 的 shared encoder / soft code distribution 可选分支与 SwanLab 记录。
- M6: Granularity-Aware Hypergraph Expansion 按当前计划先跳过，作为后续 optional branch。
- LLM query span decomposition + segment retrieval 当前按计划保留为后续可选项。
