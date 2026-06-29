# CapabilityRQ / SkillRQ 执行计划

## 1. 目标与范围

当前项目已经从单一 **Skill Recommendation** 扩展为统一的 **Agent Capability Recommendation**。Agent 可调用的 capability 既可以是高层封装的 `skills`，也可以是更细粒度的 `tools` / `APIs` / `functions`。

后续实验采用双层设置：

- 主实验：Tool/API-level Capability Recommendation，主要使用 ToolBench 和 API-Bank。
- 副实验：Skill-level Capability Recommendation，继续使用 SkillRet、SkillRouter、SRA-Bench、SkillsBench。

CapabilityRQ / SkillRQ 的目标是构建一个面向 capability selection 的可解释残差语义量化框架：

```text
full query
-> latent capability semantic code paths
-> code-based capability retrieval
-> hypergraph implicit capability expansion
-> role-aware reranking
-> tool/skill planning support
```

本计划先交付可复现实验系统，而不是直接追求端到端 agent 执行平台。第一版重点验证以下研究问题：

| 编号 | 问题 | 核心比较 |
|---|---|---|
| RQ1 | CapabilityRQ 是否优于 full-query retrieval 和 query span decomposition | Full-query retrieval / LLM span decomposition / CapabilityRQ |
| RQ2 | 可解释 codebook 是否学到 capability semantic structure | Ordinary RQ / RQ-KMeans / CapabilityRQ |
| RQ3 | Coverage supervision 是否减少重复 code path | w/o coverage / with coverage |
| RQ4 | Hypergraph 是否补齐隐含协作 tools / APIs | CapabilityRQ only / pairwise graph / hypergraph / random expansion |
| RQ5 | Code support 是否帮助 LLM 编排 tools / skills | flat list / role / code support / code + role |
| RQ6 | 该方法是否同时适用于 tool-level 与 skill-level recommendation | ToolBench/API-Bank 主实验 + SkillRet/SkillRouter/SRA-Bench 副实验 |

## 2. 数据接入计划

历史 skill 数据仍从 `/Users/sihan/code/skill-rec/data/raw` 读取。新增 tool/API 数据从 `/Users/sihan/code/SkillRQ/data/raw` 读取。项目内部只保存规范化产物、索引、缓存和实验输出。

本项目采用分阶段数据集策略，而不是把所有数据集混在同一个训练目标里。这样能让每个实验问题有清楚的证据来源，也能避免早期工程复杂度失控。

### 2.1 ToolBench

路径：

```text
/Users/sihan/code/SkillRQ/data/raw/ToolBench/data/
```

用途：

- 主实验数据集，用于大规模 tool/API recommendation。
- 从 `toolenv/tools/**.json` 抽取 tool/API capability objects。
- 从 `instruction/*.json` 和 `test_instruction/*.json` 抽取 query、available APIs、gold API set。
- 从 `answer/*_answer/*.json` 可选抽取 tool call trajectory、intermediate observations 和 final answer。
- 用成功 tool-use trajectories 构建 tool co-occurrence hypergraph：节点为 tools/APIs，超边为一次任务共同使用的一组 tools/APIs。

### 2.2 API-Bank

路径：

```text
/Users/sihan/code/SkillRQ/data/raw/DAMO-ConvAI/api-bank/
```

用途：

- 补充验证集，用于多轮 tool-use、API 调用链、工具依赖和较小规模工具库的泛化评估。
- 从 `data/all_apis.csv` 抽取 API capability objects。
- 从 `lv1-lv2-samples/**/*.jsonl` 抽取 dialogue、gold API set、tool call sequence、tool arguments、observations 和 final answer。

### 2.3 SkillRet

路径：

```text
/Users/sihan/code/skill-rec/data/raw/skillret/
```

用途：

- 副实验数据集，用于验证方法在高层 skill library 上的泛化能力。
- SkillRet 单个 query 平均 gold skills 约为 `1.99`，高阶协作关系较弱，因此 skill-level recommendation 默认关闭 hypergraph 分支，仅作为可选消融。
- 已完成 SkillRet canonical normalization，产物位于 `data/processed/`。

### 2.4 SkillRouter / SRA-Bench / SkillsBench

用途：

- SkillRouter：hard retrieval 与 skill body evidence 消融。
- SRA-Bench：skill-level coverage supervision / hypergraph 可选消融。
- SkillsBench：最终 Agent 端到端 skill-level 验证，当前不下载，不阻塞主实验。

## 3. 标准 Capability Recommendation 数据格式

ToolBench 与 API-Bank 需要统一转换为以下标准文件：

```text
data/processed/capability/
  capabilities.jsonl
  capability_queries.jsonl
  capability_qrels.jsonl
  capability_sequences.jsonl
  capability_stats.json
```

### `capabilities.jsonl`

每行表示一个 tool/API/function/skill capability object。

核心字段：

- `capability_id`
- `source_dataset`
- `source_capability_id`
- `capability_type`
- `name`
- `description`
- `category`
- `domain`
- `provider`
- `tool_name`
- `api_name`
- `api_schema`
- `parameters`
- `required_parameters`
- `optional_parameters`
- `input_schema`
- `output_schema`
- `method`
- `endpoint`
- `raw`

### `capability_queries.jsonl`

每行表示一条 user instruction、dialogue 或 task。

核心字段：

- `query_id`
- `source_dataset`
- `source_query_id`
- `source_split`
- `query`
- `gold_capability_ids`
- `available_capability_ids`
- `tool_call_sequence`
- `tool_calls_per_trajectory`
- `unique_tools_per_query`
- `tool_arguments`
- `intermediate_observations`
- `final_answer`
- `success`
- `raw`

需要特别区分：

- `unique_tools_per_query`：用于 tool recommendation / set selection。
- `tool_calls_per_trajectory`：用于 tool sequence / execution order 分析。

### `capability_qrels.jsonl`

每行表示 query 与 gold capability 的 relevance 标注。

字段：

- `query_id`
- `capability_id`
- `relevance`
- `source_dataset`
- `source_split`

### `capability_sequences.jsonl`

每行表示一次 trajectory 中的一个 tool/API call。

字段：

- `query_id`
- `step_index`
- `capability_id`
- `arguments`
- `observation`
- `source_dataset`
- `source_split`

当前已新增转换命令：

```bash
python3 -m skillrq capability build --dataset api_bank
python3 -m skillrq capability build --dataset toolbench --skip-answer-trees --limit-tools 100 --limit-queries 50
python3 -m skillrq capability build --dataset all --skip-answer-trees --limit-tools 100 --limit-queries 50
```

全量 ToolBench 转换不传 `--limit-tools` / `--limit-queries`；若要抽取 answer tree trajectory，不传 `--skip-answer-trees`。

## 4. 数据集使用评价与阶段路线

### Phase 1: ToolBench 训练 CapabilityRQ 主体

ToolBench 是主实验数据集，用于训练和评估：

- query-to-tool retrieval
- CapabilityRQ codebook
- query-to-code prediction
- multi-tool coverage
- hypergraph-based implicit tool expansion
- execution-stage modeling

重点指标：

- Recall@K
- NDCG@K
- Completeness@K
- Tool Set Recall@K
- Kendall-tau
- Transition Accuracy
- First-Tool Accuracy
- Task Success Rate

### Phase 2: API-Bank 补充泛化验证

API-Bank 用于验证 CapabilityRQ 在不同数据分布下是否保持稳定：

- tool recall
- tool sequence ordering
- API dependency modeling
- multi-turn tool-use robustness
- end-to-end task success

### Phase 3: SkillRet / SkillRouter 副实验

SkillRet 和 SkillRouter 用于 skill-level capability recommendation 的副实验，验证方法在高层 skill library 上是否仍优于：

- full-query dense retrieval
- LLM query span decomposition
- RQ-KMeans
- ordinary RQ-VAE

SkillRet 默认关闭 hypergraph 分支，仅作为可选消融。

### Phase 4: SkillsBench 最终 Agent 端到端验证

SkillsBench 不建议作为早期训练依赖。它更适合作为最终验证集，回答“推荐出来的 skills 是否真的帮助 Agent 完成任务”。

主要比较：

- No Skills
- Flat Top-k Skills
- LLM Span Decomposition + Skills
- CapabilityRQ / SkillRQ Skills
- CapabilityRQ / SkillRQ + Role Support
- CapabilityRQ / SkillRQ + Hypergraph Expansion
- Oracle / Curated Skills

## 5. 标准化数据产物

SkillRet 产物仍保留在：

```text
data/processed/
  skills.jsonl
  queries.jsonl
  qrels.jsonl
  task_skill_sets.jsonl
  roles.jsonl
  splits/
    train.jsonl
    dev.jsonl
    test.jsonl
```

建议 schema：

```json
{
  "skill_id": "string",
  "name": "string",
  "namespace": "string",
  "description": "string",
  "body": "string",
  "domain_label": "string",
  "operation_label": "string",
  "source_dataset": "skillret|skillrouter|sra_bench"
}
```

```json
{
  "query_id": "string",
  "query": "string",
  "gold_skill_ids": ["string"],
  "source_dataset": "string",
  "difficulty": "optional string",
  "domain": "optional string"
}
```

Capability 产物新增在：

```text
data/processed/capability/
```

## 6. MVP 里程碑

### M0: 项目初始化 [done]

交付物：

- Python package skeleton。已创建 `skillrq/`、`skillrq/__main__.py`、`skillrq/cli.py`。
- config system。已创建 `skillrq/config/`，支持读取 `configs/paths.yaml` 的 flat YAML 路径配置。
- data path convention。已创建 `configs/paths.yaml`，并建立 `data/processed`、`data/indexes`、`data/cache`、`runs`、`reports` 目录约定。
- smoke test。已创建 `tests/test_cli.py` 和 `tests/test_config.py`。

验收：

- `python3 -m skillrq --help` 可运行。
- `/opt/homebrew/anaconda3/bin/pytest` 通过，结果为 `3 passed`。

新增 CapabilityRQ M0 扩展项 [done]:

- 配置新增 `capability_raw_root=data/raw` 与 `capability_processed_root=data/processed/capability`。
- CLI 新增 `python3 -m skillrq capability build`。
- 脚本新增 `scripts/build_capability_data.py`。
- 报告脚本新增 `scripts/write_capability_data_report.py`。
- 测试新增 capability fixture normalization。
- 当前 smoke 验证：`/opt/homebrew/anaconda3/bin/pytest` 通过，结果为 `5 passed`。

### M1: 数据规范化 [done]

任务：

- 优先实现 SkillRet loader，产出 Phase 1 可用的 canonical files。已完成 `skillrq/data/loaders/skillret.py`。
- 预留 SkillRouter、SRA-Bench loader 接口，但不阻塞 Phase 1。已完成 placeholder loader。
- 合并 skill library，处理 ID 冲突。已按 `skill_id` 去重合并 train/test skills，并保留 `source_dataset`、`source_split`。
- 输出 canonical processed files。已完成 `python3 -m skillrq data build --dataset skillret`。

验收：

- 能统计 skills、queries、qrels、multi-skill query 数量。真实 SkillRet 构建结果：`skills=16783`、`queries=68256`、`qrels=135537`、`multi_skill_queries=45161`。
- 每条 query 至少能解析出 `gold_skill_ids`。真实构建结果：`queries_without_gold_skills=0`。
- 原始数据不被修改。loader 只读 `/Users/sihan/code/skill-rec/data/raw/skillret`，产物写入 `data/processed/`。
- qrel 引用完整性通过。真实构建结果：`missing_qrel_query_ids=0`、`missing_qrel_skill_ids=0`。
- `/opt/homebrew/anaconda3/bin/pytest` 通过，结果为 `4 passed`。

产物：

- `data/processed/skills.jsonl`
- `data/processed/queries.jsonl`
- `data/processed/qrels.jsonl`
- `data/processed/task_skill_sets.jsonl`
- `data/processed/roles.jsonl`
- `data/processed/splits/train.jsonl`
- `data/processed/splits/dev.jsonl`
- `data/processed/splits/test.jsonl`
- `data/processed/stats.json`

新增 CapabilityRQ M1 扩展项 [done]:

- 已新增 API-Bank 转换器：`skillrq/capability/loaders/api_bank.py`。
- 已新增 ToolBench 转换器：`skillrq/capability/loaders/toolbench.py`。
- 已新增标准 capability schema：`skillrq/capability/schema.py`。
- 已新增流式 ToolBench answer tree 抽取，避免全量 trajectory 一次性驻留内存。
- ToolBench capability 的 `raw` 字段已改为轻量 provenance 摘要，API 结构信息保留在 `api_schema`、`parameters`、`input_schema`、`output_schema` 等标准字段中。
- API-Bank 全量 smoke 已通过：
  - `capabilities=101`
  - `queries=263`
  - `qrels=493`
  - `sequences=508`
  - `avg_unique_tools_per_query=1.8745`
  - `avg_tool_calls_per_trajectory=1.9316`
- ToolBench 采样 smoke 已通过：
  - 命令：`python3 -m skillrq capability build --dataset toolbench --skip-answer-trees --limit-tools 100 --limit-queries 50`
  - `capabilities=129`
  - `queries=50`
  - `qrels=96`
  - `avg_unique_tools_per_query=1.92`
- ToolBench + API-Bank 全量正式构建已完成：
  - 命令：`python3 -m skillrq capability build --dataset all`
  - 输出目录：`data/processed/capability/`
  - `capabilities=64645`
  - `queries=329625`
  - `qrels=813490`
  - `sequences=447066`
  - `queries_by_dataset={api_bank: 263, toolbench: 329362}`
  - `min_unique_tools_per_query=0`
  - `max_unique_tools_per_query=10`
  - `avg_unique_tools_per_query=2.4679`
  - `min_tool_calls_per_trajectory=0`
  - `max_tool_calls_per_trajectory=5`
  - `avg_tool_calls_per_trajectory=1.3563`
- 已生成 capability 数据报告：`docs/CAPABILITY_DATA_REPORT.md`。

### M2: Capability Retrieval Baselines [done]

任务：

* 以 ToolBench / API-Bank 为主实验数据，实现 tool/API-level full-query retrieval baseline。
* 保留 SkillRet / SkillRouter 的 skill-level retrieval baseline 作为副实验。
* 实现统一 capability retrieval 框架，使 `tools`、`APIs`、`functions`、`skills` 都能被表示为 capability object。
* 实现至少两类 baseline：

  * BM25 / lexical retrieval；
  * Dense retrieval，当前实现为本地可复现的 hashing dense baseline；
  * 可选：LLM query span decomposition + segment retrieval。按当前阶段安排先保留，不写入 M2 交付。
* 建立统一 metrics 计算模块，支持 tool-level 与 skill-level 两套指标。
* 明确区分：

  * `unique tools per query`：用于 tool set recommendation；
  * `tool calls per trajectory`：用于 tool sequence / execution-order evaluation。

主实验数据：

* ToolBench；
* API-Bank。

副实验数据：

* SkillRet；
* SkillRouter。

比较方法：

```text
Tool/API-level:
  BM25 retrieval
  Full-query dense retrieval
  LLM span decomposition + retrieval [deferred]
  Optional: ToolBench / API-Bank official retriever if available

Skill-level:
  BM25 retrieval
  Full-query dense retrieval
  LLM span decomposition + retrieval [deferred]
```

实现状态：

* 已新增统一 retrieval 数据结构：`skillrq/retrieval/types.py`。
* 已新增数据适配器：`skillrq/retrieval/datasets.py`，支持 ToolBench、API-Bank、SkillRet、SkillRouter。
* 已新增 BM25 baseline：`skillrq/retrieval/bm25.py`。
* 已新增 hashing dense baseline：`skillrq/retrieval/dense.py`。
* 已新增统一 metrics：`skillrq/retrieval/metrics.py`。
* 已新增数据统计报告生成：`skillrq/retrieval/data_stats.py`。
* 已新增 M2 runner：`skillrq/retrieval/runner.py`。
* CLI 已新增：

```bash
python3 -m skillrq m2 run
```

默认评估规模：

* `--max-queries 300`
* `--max-candidates 10000`
* gold candidates 始终强制保留，不会因候选截断丢失 gold label。
* `--max-queries 0 --max-candidates 0` 可运行全量查询/候选，但 ToolBench 与 SkillRouter 会显著更慢。

验收：

* 能在 ToolBench / API-Bank 上输出：

  * Recall@K；
  * NDCG@K；
  * MRR@K；
  * Completeness@K；
  * Tool Set Recall@K；
  * First-Tool Accuracy；
  * Transition Accuracy；
  * Kendall-tau。
* 能在 SkillRet / SkillRouter 上输出：

  * Recall@K；
  * NDCG@K；
  * MRR@K；
  * Completeness@K；
  * Skill Set Recall@K。
* 能生成 baseline run artifact：

```text
runs/
  m2_baseline_retrieval/
    toolbench/
    api_bank/
    skillret/
    skillrouter/
```

* 能生成数据统计报告：

```text
reports/data_stats/
  toolbench_stats.json
  api_bank_stats.json
  skillret_stats.json
  skillrouter_stats.json
```

统计内容至少包括：

```text
query_count
capability_count
avg_unique_capabilities_per_query
avg_tool_calls_per_trajectory
multi_capability_query_ratio
unique_capabilities_ge_3_ratio
unique_capabilities_ge_5_ratio
```

本次完成记录：

* 已执行正式命令：`python3 -m skillrq m2 run`。
* 已生成 summary：`runs/m2_baseline_retrieval/summary.json`。
* 已生成每个 dataset/method 的 `metrics.json`、`predictions.jsonl`、`run_config.json`：
  * `runs/m2_baseline_retrieval/toolbench/{bm25,dense}/`
  * `runs/m2_baseline_retrieval/api_bank/{bm25,dense}/`
  * `runs/m2_baseline_retrieval/skillret/{bm25,dense}/`
  * `runs/m2_baseline_retrieval/skillrouter/{bm25,dense}/`
* 已生成数据统计报告：
  * `reports/data_stats/toolbench_stats.json`
  * `reports/data_stats/api_bank_stats.json`
  * `reports/data_stats/skillret_stats.json`
  * `reports/data_stats/skillrouter_stats.json`
* 本次正式 run 覆盖：
  * ToolBench：`run_query_count=300`，`run_candidate_count=10302`，其中 `sequence_query_count=150`。
  * API-Bank：`run_query_count=261`，`run_candidate_count=101`，其中 `sequence_query_count=261`。
  * SkillRet：`run_query_count=300`，`run_candidate_count=10219`。
  * SkillRouter：`run_query_count=85`，`run_candidate_count=10192`。
* 测试：`/opt/homebrew/anaconda3/bin/pytest` 通过，结果为 `8 passed`。

---

### M3: CapabilityRQ Codebook v1 [done]

任务：

* 将原来的 SkillRQ codebook 扩展为 CapabilityRQ codebook，使其同时支持 tool/API/function/skill。
* 主实验优先基于 ToolBench / API-Bank 学习 tool/API-level capability semantic codes。
* 副实验在 SkillRet / SkillRouter 上验证同一套 codebook 设计是否能迁移到 skill-level capability。
* 多层 codebook 设计保持统一：

```text
L1: Domain / Scenario / Artifact
L2: Operation / API Function / Capability
L3: Role / Execution Stage
L4: Input-Output / Constraint / Implementation Detail
```

Tool/API-level codebook 设计：

* L1 从 ToolBench category、API-Bank domain、tool provider 或 API category 初始化；
* L2 从 tool name、API name、description、endpoint、schema、function description 中学习；
* L3 从 tool call sequence 中弱监督得到，例如：

  * START；
  * SUPPORT；
  * CHECK；
  * FINALIZE；
  * FALLBACK；
* L4 从 parameters、input schema、output schema、required fields、method、endpoint、constraints 中学习。

Skill-level codebook 设计：

* L1 从 SkillRet / SkillRouter category 初始化；
* L2 从 skill body 学习；
* L3 使用 START / SUPPORT / CHECK / AVOID 等弱标签；
* L4 从 skill body 中的 constraints、examples、warnings、limitations 学习。

对照方法：

```text
KMeans [reference only, not reproduced in M3]
RQ-KMeans [reference only, not reproduced in M3]
Ordinary RQ-VAE [reference only, not reproduced in M3]
CapabilityRQ
```

实现状态：

* 已新增 CapabilityRQ codebook package：`skillrq/codebook/`。
* 已新增四层 code assignment 逻辑：`skillrq/codebook/assign.py`。
* 已新增稳定 code ID 与 semantic ID 生成：`skillrq/codebook/ids.py`。
* 已新增 code quality 指标：`skillrq/codebook/quality.py`。
* 已新增 Code Card 生成：`skillrq/codebook/cards.py`。
* 已新增 M3 runner：`skillrq/codebook/runner.py`。
* CLI 已新增：

```bash
python3 -m skillrq m3 build
```

当前 M3 v1 为启发式、可解释、可复现 codebook：

* L1 使用 category、domain、provider、SkillRet domain label、SkillRouter skill source 等 domain/scenario 信号；
* L2 使用 tool name、API name、skill primary action、skill name 等 operation 信号；
* L3 使用 tool call sequence 弱监督得到 START / SUPPORT / FINALIZE，并用文本规则补充 CHECK / AVOID 等角色；
* L4 使用 parameters、input/output schema、method、constraints、examples、validation 等 IO/detail 信号。

验收：

* 每个 tool/API/function/skill capability 都有统一 semantic code path：

```text
capability -> [L1, L2, L3, L4]
```

* ToolBench / API-Bank 中每个 tool/API 都能生成 semantic ID。
* SkillRet / SkillRouter 中每个 skill 也能生成 semantic ID。
* 输出 code quality 指标：

  * Code Purity；
  * Code Usage Entropy；
  * Category Alignment；
  * Role Alignment；
  * Code Collapse Rate。
* 生成 code assignments：

```text
data/processed/capability/code_assignments.jsonl
data/processed/skill/code_assignments.jsonl
```

* 生成 Code Card 草稿：

```text
reports/code_cards/
  toolbench_code_cards.md
  api_bank_code_cards.md
  skillret_code_cards.md
```

本次完成记录：

* 已执行正式命令：`python3 -m skillrq m3 build`。
* 已生成 capability code assignments：
  * `data/processed/capability/code_assignments.jsonl`
  * 行数：`64645`
* 已生成 skill code assignments：
  * `data/processed/skill/code_assignments.jsonl`
  * 行数：`95924`
  * 其中包含 SkillRet 与去重后的 SkillRouter unique skill IDs。
* 已生成 code quality：
  * `data/processed/capability/code_quality.json`
  * `data/processed/skill/code_quality.json`
  * `reports/code_cards/m3_codebook_summary.json`
* 已生成 Code Card 草稿：
  * `reports/code_cards/toolbench_code_cards.md`
  * `reports/code_cards/api_bank_code_cards.md`
  * `reports/code_cards/skillret_code_cards.md`
* 全量 assignment 统计：
  * `capability_assignments=64645`
  * `skill_assignments=95924`
  * `total_assignments=160569`
  * `unique_semantic_code_paths=124841`
  * `code_usage_entropy=0.9705`
  * `code_collapse_rate=0.0022`
* 测试：`/opt/homebrew/anaconda3/bin/pytest` 通过，结果为 `9 passed`。

---

### M4: Query-to-Code Latent Capability Decomposition [implementation done, cloud training ready]

任务：

* 将原来的 Query Latent Skill Decomposition 更新为 Query-to-Code Latent Capability Decomposition。
* 主实验在 ToolBench / API-Bank 上训练 query / instruction 到 tool semantic code paths 的映射。
* 副实验在 SkillRet 上训练 query 到 skill semantic code paths 的映射。
* 不再默认进行 query span decomposition，而是让完整 query 预测多个 capability semantic code paths。
* 对每条 query 输出 top-N code paths，并通过 code path 检索 explicit capability candidates。

Tool/API-level 输入：

```text
instruction / dialogue
available tools / APIs if provided
tool schema text
gold tool set
tool call sequence
```

Skill-level 输入：

```text
query
skill name / description / body
gold skill set
```

输出：

```json
{
  "query_id": "...",
  "predicted_code_paths": [
    {
      "path_id": "P1",
      "codes": ["L1-xx", "L2-xx", "L3-xx", "L4-xx"],
      "role_hint": "START",
      "score": 0.0
    }
  ],
  "retrieved_capabilities": [...]
}
```

比较方法：

```text
Full-query dense retrieval
LLM span decomposition + retrieval [interface reserved, not required for current training run]
RQ-KMeans code retrieval
Ordinary RQ-VAE code retrieval
CapabilityRQ code retrieval
```

实现状态：

* 已新增 M4 package：`skillrq/m4/`。
* 已新增 M4 监督数据准备：`skillrq/m4/data.py`。
* 已新增 PyTorch query-to-code 主模型：`skillrq/m4/model.py`。
* 已新增 CapabilityRQ 训练入口：`skillrq/m4/train.py`。
* 已新增 code path 预测与 explicit candidate retrieval：`skillrq/m4/predict.py`。
* 已新增 M4 评估：`skillrq/m4/evaluate.py`。
* 已新增 RQ-KMeans 与 Ordinary RQ-VAE 对照训练入口：`skillrq/m4/baselines.py`。
* 已新增 PyTorch optional dependency：`pyproject.toml` 中的 `train` extra。
* CLI 已新增：

```bash
python3 -m skillrq m4 prepare
python3 -m skillrq m4 train
python3 -m skillrq m4 predict
python3 -m skillrq m4 evaluate
python3 -m skillrq m4 rq-kmeans
python3 -m skillrq m4 rq-vae
```

训练设计：

* CapabilityRQ 主模型将每条 query 与 gold capability/skill 的 semantic code path 配对训练。
* 模型结构为 PyTorch EmbeddingBag query encoder + MLP + 四个 code-level classification heads。
* 训练目标为 L1/L2/L3/L4 四层 cross entropy。
* 推理时使用各层 top codes 的 beam combination 生成 top-N code paths。
* candidate retrieval 使用 predicted code path 与 candidate code path 的层级匹配分数，并输出 compact evidence。
* RQ-KMeans 对照使用 PyTorch 上的 residual k-means，在 candidate text hashed vectors 上学习多层 codebook。
* Ordinary RQ-VAE 对照使用 PyTorch encoder / residual vector quantization / decoder reconstruction objective。

验收：

* 每条 ToolBench / API-Bank query 输出 top-N tool/API code paths。
* 每条 SkillRet query 输出 top-N skill code paths。
* 每个候选 capability 附带：

  * matched code path；
  * code match score；
  * code explanation；
  * capability text evidence。
* 对比 full-query retrieval，至少在以下一个维度有提升：

  * Recall@K；
  * Completeness@K；
  * Tool Set Recall@K；
  * Candidate Pool Size；
  * Recall under Same Candidate Budget。

本次完成记录：

* 已生成 capability-level M4 训练数据：
  * 命令：`python3 -m skillrq m4 prepare --target capability --datasets toolbench api_bank`
  * 输出目录：`data/processed/m4/capability/`
  * `candidates=64645`
  * `queries=329207`
  * `train_pairs=813445`
  * `test_queries=1100`
  * `avg_gold_code_paths_per_query=2.4271`
* 已生成 skill-level M4 训练数据：
  * 命令：`python3 -m skillrq m4 prepare --target skill --datasets skillret`
  * 输出目录：`data/processed/m4/skill/`
  * `candidates=16783`
  * `queries=68256`
  * `train_pairs=135537`
  * `test_queries=4997`
  * `avg_gold_code_paths_per_query=1.9806`
* README 已写入云服务器训练命令：
  * CapabilityRQ capability/skill training；
  * CapabilityRQ prediction；
  * M4 evaluation；
  * RQ-KMeans training；
  * Ordinary RQ-VAE training。
* 本地环境未安装 PyTorch，未在本机训练 checkpoint；训练代码已按云服务器执行路径准备。
* 测试：`/opt/homebrew/anaconda3/bin/pytest` 通过，结果为 `10 passed`。

---

### M5: Residual Multi-Code Path Selector with Coverage Supervision [已完成工程实现，待云端训练]

目标：

* 将 M4 的 one-shot query-to-code prediction 扩展为 residual multi-code path selection。
* 每一步根据 `query + residual_state` 预测下一个 code path，使新路径优先覆盖尚未覆盖的 gold tools / skills。
* 显式学习 `coverage_gain`，避免多个 code paths 重复解释同一个 dominant capability。
* Tool/API-level 作为主实验；SkillRet skill-level 作为副实验和 anti-redundancy 验证。

核心流程：

```text
query q
covered_0 = empty

for step t in 1..T:
  residual_state_t = summarize(covered_{t-1})
  path_t, coverage_gain_t = selector(q, residual_state_t)
  candidates_t = retrieve_by_code_path(path_t)
  covered_t = covered_{t-1} union candidates_t
```

训练监督构造：

* 输入：M4 的 `queries.jsonl` 与 `candidates.jsonl`。
* 对每条 query 的 gold capability set，按 candidate semantic code path 分组。
* 用 greedy residual oracle 选择每一步 coverage gain 最大的 code path。
* 每个训练样本包含：

```text
query_id
query
step_index
residual_state
target_ids
semantic_id
code_path
role_hint
coverage_gain
normalized_coverage_gain
covered_before
remaining_after
```

模型与损失：

```text
L_m5 = CE(l1) + CE(l2) + CE(l3) + CE(l4)
     + coverage_weight * MSE(sigmoid(predicted_coverage_gain), normalized_coverage_gain)
```

SwanLab 记录：

```text
train/loss
train/code_loss
train/coverage_loss
dev/loss
dev/code_loss
dev/coverage_loss
dev/l1_accuracy
dev/l2_accuracy
dev/l3_accuracy
dev/l4_accuracy
dev/path_exact_match
predict/queries
predict/avg_steps
```

推理策略：

* 逐步预测 residual code path，而不是一次性输出 flat top-k。
* 每一步对重复 `semantic_id` 做跳过。
* 检索候选时加入 novelty bonus，已覆盖候选会受到惩罚。
* 当 gold set 已全部覆盖或达到 `max_steps` 时停止。

评估指标：

```text
Step-wise Coverage Gain
Redundant Code Path Ratio
Candidate Redundancy Ratio
Recall@K
Completeness@K
Tool Set Recall@K / Skill Set Recall@K
```

新增命令：

```bash
python3 -m skillrq m5 prepare --target capability
python3 -m skillrq m5 prepare --target skill

python3 -m skillrq m5 train --target capability --device cuda
python3 -m skillrq m5 predict --target capability --checkpoint-root runs/m5_residual_selector/capability --device cuda
python3 -m skillrq m5 evaluate --prediction-path runs/m5_residual_selector/predictions/capability/predictions.jsonl --output-path reports/tables/m5_coverage_supervision_capability.json
```

代码与产物：

```text
skillrq/m5/data.py
skillrq/m5/model.py
skillrq/m5/train.py
skillrq/m5/predict.py
skillrq/m5/evaluate.py

data/processed/m5/capability/residual_examples.jsonl
data/processed/m5/capability/query_residual_plans.jsonl
data/processed/m5/capability/stats.json
data/processed/m5/skill/residual_examples.jsonl
data/processed/m5/skill/query_residual_plans.jsonl
data/processed/m5/skill/stats.json
```

本次完成记录：

* 已生成 capability-level M5 coverage supervision 数据：
  * 输出目录：`data/processed/m5/capability/`
  * `queries=329207`
  * `residual_examples=798736`
  * `avg_steps_per_query=2.4262`
  * `avg_coverage_gain=1.0180`
* 已生成 skill-level M5 coverage supervision 数据：
  * 输出目录：`data/processed/m5/skill/`
  * `queries=68256`
  * `residual_examples=135185`
  * `avg_steps_per_query=1.9806`
  * `avg_coverage_gain=1.0026`
* 已实现 M5 CLI：`prepare`、`train`、`predict`、`evaluate`。
* 已实现 M5 单元测试，覆盖 residual oracle 构造与 coverage metrics。
* 本地环境未安装 PyTorch，未训练 checkpoint；训练命令已写入 README，供云服务器执行。
* 测试：`/opt/homebrew/anaconda3/bin/pytest` 通过，结果为 `12 passed`。

---

### M6: Granularity-Aware Hypergraph Expansion

任务：

* 将 Hypergraph 从 skill-only 分支更新为 granularity-aware capability hypergraph。
* 主实验：在 ToolBench / API-Bank 上默认启用 hypergraph branch。
* 副实验：在 SkillRet / SkillRouter 上默认关闭 hypergraph branch，仅作为 optional ablation。
* 构建 tool/API co-occurrence hypergraph：

  * 节点：tools / APIs / functions；
  * 超边：一次成功任务中共同使用的一组 unique tools / APIs；
  * 超边权重：共现频次、任务成功率、PMI / Lift、角色组合稳定性、transition confidence。
* 构建 skill co-occurrence hypergraph：

  * 仅在 SRA-Bench / SkillsBench 或具有 task-level multi-skill set 的数据上启用；
  * SkillRet 中由于 avg gold skills 较低，不作为 hypergraph 主训练来源。

Tool/API-level hypergraph 用途：

```text
CapabilityRQ explicit tool candidates
-> hypergraph expansion
-> implicit SUPPORT / CHECK / FALLBACK tool candidates
-> reranker internal graph prior
```

Skill-level hypergraph 用途：

```text
SkillRQ explicit skill candidates
-> 默认不扩展
-> optional: pairwise graph / hypergraph ablation
```

Granularity-aware branch decision：

```text
if capability_type == "tool" or "api":
    enable hypergraph expansion by default
elif capability_type == "skill":
    disable hypergraph expansion by default
```

可选增强：

```text
enable hypergraph if estimated_capability_count(q) >= threshold
enable hypergraph if role coverage is incomplete
enable hypergraph if high-confidence hyperedge exists
```

对照方法：

```text
CapabilityRQ only
CapabilityRQ + pairwise graph
CapabilityRQ + hypergraph
CapabilityRQ + random expansion
```

验收：

* ToolBench / API-Bank 上输出：

  * hyperedges；
  * expanded tool candidates；
  * hypergraph support score；
  * implicit tool recall。
* SkillRet / SkillRouter 上输出：

  * no-hypergraph default result；
  * hypergraph optional ablation result。
* 指标包括：

  * Implicit Tool Recall@K；
  * Support Tool Recall@K；
  * Check Tool Recall@K；
  * Tool Set Recall@K；
  * Noise Ratio；
  * Generic Tool Ratio；
  * Redundant Capability Ratio。
* 输出：

```text
data/processed/capability/hypergraph.jsonl
reports/tables/m6_hypergraph_expansion_tool.csv
reports/tables/m6_hypergraph_ablation_skill.csv
```

---

### M7: Role-Aware and Sequence-Aware Reranker [已完成工程实现，待云端训练/推理]

本阶段按当前要求跳过 M6 hypergraph optional branch。M7 保留 `hypergraph_support_score` 特征位，但默认值为 `0.0`，不依赖 M6 产物。

目标：

* 将 M4 / M5 产生的 candidate pool 重排为更适合 Agent planner 使用的 capability list。
* 同时输出 relevance score、suggested role、execution stage、optional order score 和 compact support evidence。
* Tool/API-level 主实验支持 tool set reranking 与 execution-order prediction。
* Skill-level 副实验支持 skill reranking 与 body evidence / role support 输出。

输入与输出：

```text
输入训练数据:
data/processed/m4/{target}/queries.jsonl
data/processed/m4/{target}/candidates.jsonl

输入推理数据:
M4 predictions.jsonl 或 M5 predictions.jsonl

输出:
runs/m7_reranker/predictions/{target}/reranked_predictions.jsonl
```

Reranker 输出字段：

```text
candidate_id
name
semantic_id
matched_code_path
suggested_role
execution_stage
relevance_score
optional_order_score
final_score
features
compact_support_evidence
code_explanation
```

训练监督构造：

* 正样本：每条 query 的 gold tools / skills。
* 负样本：同 source / 同 L1 code path 下采样 hard negatives。
* role label：来自 M3/M4 codebook 中的 `role_hint`。
* stage label：由 `sequence_ids` 映射为 `FIRST / MIDDLE / FINAL / CHECK / AVOID`。
* order score：`1 / (sequence_position + 1)`，用于学习 first-tool 与后续执行顺序。

特征：

```text
code_match_score
matched_levels
text_overlap_score
schema_evidence_score
parameter_compatibility_score
role_compatibility_score
coverage_gain_score
hypergraph_support_score  # 当前 M6 跳过，默认 0.0
first_tool_prior
transition_prior
redundancy_penalty
generic_penalty
constraint_violation_penalty
```

模型与损失：

```text
input = query text + candidate text + numeric features

L_m7 = BCE(relevance)
     + role_weight * CE(role)
     + stage_weight * CE(stage)
     + order_weight * MSE(sigmoid(order_score), target_order_score)
```

可选 joint ablation 分支，默认关闭：

```text
Branch A: shared query encoder + code encoder
  flag: --enable-shared-encoder
  默认: off
  含义: residual code prediction head 与 reranker head 共用底层 query representation，
        使 reranker loss 可以更新同一个 query encoder。

Branch B: soft code distribution
  flag: --enable-soft-code-distribution
  默认: off
  含义: 不只使用 hard argmax code path，而是保留每层 code probability distribution，
        用 candidate code 在该分布下的 soft match score 参与 reranking。
```

joint 训练目标：

```text
L_m7_joint = BCE(relevance)
           + code_weight * [CE(l1) + CE(l2) + CE(l3) + CE(l4)]
           + role_weight * CE(role)
           + stage_weight * CE(stage)
           + order_weight * MSE(sigmoid(order_score), target_order_score)
           + soft_code_weight * L_soft_code
```

说明：

* 两个分支都不改变 M3 discrete code assignment 和 code label。
* `--enable-shared-encoder` 打开后，code prediction 与 reranking 共享 query encoder。
* `--enable-soft-code-distribution` 打开后，reranker loss 可以通过 soft code match 影响 code distribution。
* 若不打开 soft branch，`soft_code_weight` 的 effective weight 自动为 `0.0`。
* 普通 `m7 train` 仍是 offline reranker；`m7 joint-train` 用于这些消融实验。

SwanLab 记录：

```text
train/loss
train/relevance_loss
train/role_loss
train/stage_loss
train/order_loss
train/code_loss                 # joint-train
train/soft_code_loss            # joint-train
dev/relevance_accuracy
dev/role_accuracy
dev/stage_accuracy
dev/order_mse
dev/code_path_exact_match       # joint-train
predict/queries
predict/top_k
predict/avg_reranked_candidates
```

评估指标：

```text
Recall@K
NDCG@K
MRR@K
Completeness@K
Tool Set Recall@K / Skill Set Recall@K
First-Tool Accuracy
Transition Accuracy
Kendall-tau
```

新增命令：

```bash
python3 -m skillrq m7 prepare --target capability
python3 -m skillrq m7 prepare --target skill

python3 -m skillrq m7 train --target capability --device cuda
python3 -m skillrq m7 joint-train --target capability --device cuda
python3 -m skillrq m7 joint-train --target capability --enable-shared-encoder --device cuda
python3 -m skillrq m7 joint-train --target capability --enable-soft-code-distribution --device cuda
python3 -m skillrq m7 joint-train --target capability --enable-shared-encoder --enable-soft-code-distribution --device cuda
python3 -m skillrq m7 predict --target capability --prediction-path runs/m5_residual_selector/predictions/capability/predictions.jsonl --checkpoint-root runs/m7_reranker/capability --device cuda
python3 -m skillrq m7 joint-predict --target capability --prediction-path runs/m5_residual_selector/predictions/capability/predictions.jsonl --checkpoint-root runs/m7_joint_reranker/capability --device cuda
python3 -m skillrq m7 evaluate --prediction-path runs/m7_reranker/predictions/capability/reranked_predictions.jsonl --output-path reports/tables/m7_tool_reranking.json
```

代码与产物：

```text
skillrq/m7/features.py
skillrq/m7/data.py
skillrq/m7/model.py
skillrq/m7/train.py
skillrq/m7/predict.py
skillrq/m7/evaluate.py
skillrq/m7/joint_model.py
skillrq/m7/joint_train.py
skillrq/m7/joint_predict.py

data/processed/m7/capability/rerank_examples.jsonl
data/processed/m7/capability/query_candidate_pools.jsonl
data/processed/m7/capability/stats.json
data/processed/m7/skill/rerank_examples.jsonl
data/processed/m7/skill/query_candidate_pools.jsonl
data/processed/m7/skill/stats.json
```

本次完成记录：

* 已生成 capability-level M7 reranker 数据：
  * 输出目录：`data/processed/m7/capability/`
  * `queries=329207`
  * `examples=2440301`
  * `positives=813445`
  * `negatives=1626856`
  * `queries_with_sequence=126332`
* 已生成 skill-level M7 reranker 数据：
  * 输出目录：`data/processed/m7/skill/`
  * `queries=68256`
  * `examples=406611`
  * `positives=135537`
  * `negatives=271074`
  * `queries_with_sequence=0`
* 已实现 M7 CLI：`prepare`、`train`、`predict`、`joint-train`、`joint-predict`、`evaluate`。
* 已新增 joint ablation 两个默认关闭分支：
  * `--enable-shared-encoder`
  * `--enable-soft-code-distribution`
* 已实现 M7 单元测试，覆盖 hard negative 构造与 retrieval / sequence metrics。
* 本地环境未安装 PyTorch，未训练 checkpoint；训练命令已写入 README，供云服务器执行。
* 测试：`/opt/homebrew/anaconda3/bin/pytest` 通过，结果为 `14 passed`。

---

### M8: Main Experiment Matrix and Paper-Level Artifacts

任务：

* 跑完整 Tool/API-level 主实验。
* 跑 Skill-level 副实验。
* 跑 hypergraph granularity-aware 消融。
* 汇总 tables、figures、case studies。
* 将 Agent end-to-end evaluation 保留为后续 M9，不阻塞 M8 的检索、表示学习与排序实验。

Tool/API-level 主实验方法：

```text
BM25
Full-query dense retrieval
LLM query span decomposition + retrieval
RQ-KMeans semantic code retrieval
Ordinary RQ-VAE
CapabilityRQ without hypergraph
CapabilityRQ with pairwise graph
CapabilityRQ with hypergraph
CapabilityRQ with hypergraph + role-aware reranking
CapabilityRQ with hypergraph + role-aware + sequence-aware reranking
```

Skill-level 副实验方法：

```text
BM25
Full-query dense retrieval
LLM query span decomposition + retrieval
RQ-KMeans semantic code retrieval
Ordinary RQ-VAE
CapabilityRQ / SkillRQ without hypergraph
CapabilityRQ / SkillRQ with optional hypergraph ablation
```

关键实验表：

```text
Table 1: ToolBench main retrieval results
Table 2: API-Bank generalization results
Table 3: Tool sequence ordering results
Table 4: Skill-level recommendation results
Table 5: Codebook quality comparison
Table 6: Hypergraph granularity-aware ablation
Table 7: Reranker feature ablation
```

关键 case studies：

```text
Case 1: ToolBench multi-tool query where full-query retrieval misses support tool
Case 2: API-Bank multi-turn dialogue requiring ordered API calls
Case 3: SkillRet query where SkillRQ controls candidate pool better than span decomposition
Case 4: Skill-level hypergraph over-expansion failure case
```

验收：

* 产物齐全：

```text
reports/tables/*.csv
reports/figures/*
reports/cases/*.md
runs/*/config.yaml
runs/*/metrics.json
```

* 每个 RQ 有对应实验设置、指标和结论模板。
* 能支撑以下论文结论：

```text
CapabilityRQ improves tool/API recommendation in fine-grained capability libraries.
Hypergraph expansion is effective for tool-level capability completion.
For high-level skill recommendation, hypergraph should be treated as an optional branch rather than a default component.
CapabilityRQ generalizes from tools/APIs to skills.
```

---

### M9: Agent-Level End-to-End Validation

任务：

* 建立轻量 Agent evaluation harness，优先支持 ToolBench / API-Bank 的 tool/API-level end-to-end 验证。
* SkillsBench 作为 skill-level end-to-end 验证，放在第二阶段。
* 不把 argument generation 作为本文核心贡献，但需要保留 tool arguments 以支持 end-to-end replay 或 evaluation。
* 比较推荐结果对 LLM Agent 执行成功率的影响。

Tool/API-level end-to-end 比较：

```text
No retrieved tools
Flat Top-k Tools
Full-query dense retrieved tools
LLM span decomposition retrieved tools
CapabilityRQ tools
CapabilityRQ + Role Support
CapabilityRQ + Hypergraph Expansion
CapabilityRQ + Hypergraph + Sequence-aware Plan
Oracle / Gold Tools
```

Skill-level end-to-end 比较：

```text
No Skills
Flat Top-k Skills
Full-query dense retrieved skills
LLM Span Decomposition + Skills
CapabilityRQ / SkillRQ Skills
CapabilityRQ / SkillRQ + Role Support
Optional: CapabilityRQ / SkillRQ + Hypergraph Expansion
Oracle / Curated Skills
```

指标：

```text
Task Success Rate
Verifier Pass Rate
Tool Loading Accuracy
Skill Loading Accuracy
First-Tool Accuracy
Tool Sequence Match
Visible Requirement Coverage
Token Usage
Runtime
```

验收：

* 至少完成 ToolBench 或 API-Bank 的一个代表性任务子集 end-to-end evaluation。
* 至少完成 SkillsBench 的一个小规模代表性任务族 case study。
* 输出：

```text
reports/tables/m9_agent_tool_end_to_end.csv
reports/tables/m9_agent_skill_end_to_end.csv
reports/cases/m9_agent_cases.md
```

---

## 7. 训练目标落地顺序

建议先实现 capability-level staged training，再逐步组合总损失：

```text
L = L_rec
  + lambda_query_capability * L_query_capability
  + lambda_role * L_role
  + lambda_sequence * L_sequence
  + lambda_hypergraph * L_hypergraph
  + lambda_coverage * L_coverage
  + lambda_interpretability * L_interpretability
  + lambda_diversity * L_diversity
```

优先级：

1. `L_query_capability`：确保 query-to-tool / query-to-skill retrieval 有效。
2. `L_rec`：确保 codebook 表达能力。
3. `L_coverage`：解决 residual code paths 重复解释 dominant capability。
4. `L_diversity`：防止 code collapse。
5. `L_role`：学习 START / SUPPORT / CHECK / FINALIZE / FALLBACK 等执行角色。
6. `L_sequence`：仅在 tool/API-level 上优先启用，用于学习 first-tool、transition 和 execution order。
7. `L_hypergraph`：主实验在 tool/API-level 启用；skill-level 默认关闭，仅作为 optional ablation。
8. `L_interpretability`：提升 code 与 category、operation、role、constraint 的对齐。

---

## 8. 评测指标

### Tool/API Retrieval and Recommendation

```text
Recall@K
NDCG@K
MRR@K
Completeness@K
Tool Set Recall@K
Tool Set Precision@K
Tool Set F1@K
Candidate Pool Size
Recall under Same Candidate Budget
```

### Tool/API Sequence and Planning

```text
First-Tool Accuracy
Kendall-tau
Transition Accuracy
Ordered Precision
Sequence Exact Match
Tool Call Coverage
```

### Skill Retrieval and Recommendation

```text
Recall@K
NDCG@K
MRR@K
Completeness@K
Skill Set Recall@K
Coverage@K
Support Skill Recall@K
Check Skill Recall@K
Implicit Skill Recall@K
```

### Code Quality

```text
Code Purity
Code Usage Entropy
Role Alignment Accuracy
Category Alignment Accuracy
Operation Alignment Accuracy
Code Collapse Rate
Code Interpretability Score
```

### Hypergraph Quality

```text
Implicit Tool Recall@K
Support Tool Recall@K
Check Tool Recall@K
Hyperedge Hit Rate
Expansion Precision
Expansion Recall
Noise Ratio
Generic Capability Ratio
Redundant Capability Ratio
```

### Agent-Level Optional

```text
Task Success Rate
Verifier Pass Rate
Tool Loading Accuracy
Skill Loading Accuracy
Visible Requirement Coverage
Token Usage
Runtime
```

---

## 9. 风险与控制

| 风险                          | 表现                                 | 控制方式                                                            |
| --------------------------- | ---------------------------------- | --------------------------------------------------------------- |
| code collapse               | 少数 code 被过度使用                      | usage entropy、diversity loss、balanced assignment                |
| residual 不等于剩余需求            | 多个 path 重复召回同一 dominant tool/skill | coverage supervision、redundancy penalty                         |
| tool sequence 与 tool set 混淆 | tool calls 很多但 unique tools 很少     | 分开统计 unique tools per query 与 tool calls per trajectory         |
| hypergraph 引入噪声             | 高频泛化 tool 被扩展                      | PMI/Lift、generic penalty、threshold tuning                       |
| skill-level hypergraph 过拟合  | SkillRet avg gold skills 低，扩展后噪声增加 | skill-level 默认关闭 hypergraph，仅做 optional ablation                |
| role label 不足               | START/SUPPORT/CHECK 弱监督不稳定         | 从 trajectory 位置、schema 类型、LLM weak labels 构造 role pseudo labels |
| ToolBench schema 脏数据        | schema 可能是 list/set/string 而非 dict | JSON-safe writer，raw 保留，标准字段降级为空对象/列表                           |
| API-Bank 多轮结构复杂             | dialogue、API call、observation 难对齐  | 先抽取 gold API set 与 sequence，再逐步处理 arguments                     |
| argument generation 范围过大    | 审稿人关注点被转移                          | 明确本文研究 tool recommendation / ordering，不研究 argument generation   |
| 过早引入 end-to-end agent       | 环境成本干扰主体建模                         | M9 才做 end-to-end，M2-M8 先完成 offline recommendation               |

---

## 10. 推荐第一周任务

1. 完成 ToolBench 全量 instruction / test_instruction 转换，生成正式 `data/processed/capability/`。
2. 完成 API-Bank 全量转换，检查 `capabilities`、`queries`、`qrels`、`sequences` 的一致性。
3. 生成 ToolBench / API-Bank 数据统计报告，重点统计：

   * unique tools per query；
   * tool calls per trajectory；
   * multi-tool query 占比；
   * unique tools >= 3 的 query 占比；
   * unique tools >= 5 的 query 占比。
4. 实现 ToolBench / API-Bank full-query dense retrieval baseline。
5. 实现 Tool Set Recall@K、Completeness@K、First-Tool Accuracy、Transition Accuracy。
6. 保留 SkillRet dense retrieval baseline 作为副实验，不阻塞 tool-level 主线。
7. 根据统计结果决定 hypergraph branch 的启用阈值和 tool-level 子集筛选策略。
