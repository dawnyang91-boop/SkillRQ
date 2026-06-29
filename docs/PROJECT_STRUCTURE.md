# SkillRQ 项目文件架构

## 1. 总体原则

项目建议采用 Python package + config-driven experiments 的结构。代码和实验产物分离，原始数据只读，所有中间产物可重建。

当前 M0 初始化采用根目录 `skillrq/` 包，以保证在仓库根目录直接运行 `python3 -m skillrq --help`。下方 `src/skillrq/` 是后续模块扩展时的目标结构；进入 M1/M2 后可继续沿用根目录包，也可在一次明确迁移中切换到 `src` layout。

```text
SkillRQ/
  README.md
  docs/
    EXECUTION_PLAN.md
    PROJECT_STRUCTURE.md
  pyproject.toml
  configs/
  data/
  src/
  scripts/
  tests/
  runs/
  reports/
```

## 2. 建议目录树

```text
SkillRQ/
  README.md
  pyproject.toml
  .gitignore

  configs/
    paths.yaml
    model/
      encoder.yaml
      skillrq.yaml
      reranker.yaml
    data/
      skillret.yaml
      skillrouter.yaml
      sra_bench.yaml
      skillsbench.yaml
    experiments/
      baseline_skillret.yaml
      rq_kmeans_skillret.yaml
      skillrq_no_hypergraph.yaml
      skillrq_with_hypergraph.yaml
      skillrq_role_reranker.yaml
      agent_eval_skillsbench.yaml

  data/
    README.md
    raw/
      DAMO-ConvAI/
        api-bank/
      ToolBench/
        data/
    processed/
      skills.jsonl
      queries.jsonl
      qrels.jsonl
      task_skill_sets.jsonl
      roles.jsonl
      capability/
        capabilities.jsonl
        capability_queries.jsonl
        capability_qrels.jsonl
        capability_sequences.jsonl
        capability_stats.json
      splits/
        train.jsonl
        dev.jsonl
        test.jsonl
    indexes/
      dense/
      bm25/
      codebook/
      hypergraph/
    cache/
      embeddings/
      code_cards/

  docs/
    EXECUTION_PLAN.md
    PROJECT_STRUCTURE.md
    DESIGN_NOTES.md
    METRICS.md
    DATA_SCHEMA.md

  src/
    skillrq/
      __init__.py
      __main__.py
      cli.py

      config/
        __init__.py
        loader.py
        schema.py

      data/
        __init__.py
        schemas.py
        canonicalize.py
        id_mapping.py
        loaders/
          __init__.py
          skillret.py
          skillrouter.py
          sra_bench.py
          skillsbench.py
        splits.py
        stats.py

      capability/
        __init__.py
        build.py
        ids.py
        schema.py
        loaders/
          __init__.py
          api_bank.py
          toolbench.py

      encoding/
        __init__.py
        text_builder.py
        embedding_backend.py
        embed_skills.py
        embed_queries.py

      retrieval/
        __init__.py
        bm25.py
        dense.py
        candidate_pool.py
        baselines/
          __init__.py
          full_query.py
          span_decomposition.py
          rq_kmeans.py

      codebook/
        __init__.py
        levels.py
        initialization.py
        residual_quantizer.py
        skill_semantic_id.py
        code_card.py
        interpretability.py

      decomposition/
        __init__.py
        query_encoder.py
        path_selector.py
        residual_state.py
        coverage.py

      hypergraph/
        __init__.py
        builder.py
        weighting.py
        expansion.py
        priors.py

      reranking/
        __init__.py
        features.py
        role_prediction.py
        evidence.py
        scorer.py
        redundancy.py

      training/
        __init__.py
        datasets.py
        objectives.py
        trainer.py
        losses/
          __init__.py
          reconstruction.py
          query_skill.py
          role.py
          hypergraph.py
          coverage.py
          interpretability.py
          diversity.py

      evaluation/
        __init__.py
        metrics.py
        retrieval_eval.py
        coverage_eval.py
        code_quality_eval.py
        agent_eval.py
        ablations.py
        skillsbench_eval.py

      planning_support/
        __init__.py
        support_formatter.py
        skill_plan.py

      utils/
        __init__.py
        io.py
        logging.py
        random.py
        typing.py

  scripts/
    build_processed_data.py
    build_capability_data.py
    build_embeddings.py
    build_codebook.py
    build_hypergraph.py
    run_retrieval.py
    train_skillrq.py
    evaluate_run.py
    export_code_cards.py
    export_report_tables.py

  tests/
    test_data_loaders.py
    test_canonical_schema.py
    test_metrics.py
    test_codebook.py
    test_hypergraph.py
    test_reranker_features.py

  runs/
    .gitkeep

  reports/
    tables/
    figures/
    cases/
```

## 3. 模块职责

### `src/skillrq/data`

负责从 raw datasets 到 canonical schema 的转换。

关键设计：

- loader 只读取 `/Users/sihan/code/skill-rec/data/raw`。
- canonicalize 负责统一 `skill_id`、`query_id`、`gold_skill_ids`。
- `id_mapping.py` 处理跨数据集 skill name、namespace 和 ID 冲突。
- `skillsbench.py` 是远期可选 loader，只有进入最终 Agent 端到端评测时才实现和启用。

### `src/skillrq/capability`

负责 ToolBench / API-Bank 到统一 Agent Capability Recommendation schema 的转换。

关键设计：

- `capabilities.jsonl` 表示 tool/API/function/skill capability object。
- `capability_queries.jsonl` 表示 user instruction、dialogue 或 task。
- `capability_qrels.jsonl` 表示 query-capability relevance。
- `capability_sequences.jsonl` 表示 trajectory 中逐步 tool/API call。
- `unique_tools_per_query` 用于 set recommendation，`tool_calls_per_trajectory` 用于 execution order 分析。
- ToolBench 全量较大，CLI 支持 `--limit-tools` / `--limit-queries` 做 smoke test。

### `src/skillrq/encoding`

负责构造 skill/query 输入文本并生成 embedding。

建议 skill text template：

```text
Name: {name}
Namespace: {namespace}
Description: {description}
Major: {major}
Sub: {sub}
Body: {body}
```

### `src/skillrq/retrieval`

负责普通检索和 baseline。

建议包含：

- full-query dense retrieval。
- lexical/BM25 retrieval。
- query span decomposition baseline 的离线接口。
- ordinary RQ-KMeans baseline。

### `src/skillrq/codebook`

负责 SkillRQ 的核心可解释 codebook。

四层结构：

| Level | 名称 | 作用 |
|---|---|---|
| L1 | Domain / Artifact | 任务领域或处理对象 |
| L2 | Operation / Capability | skill 核心能力 |
| L3 | Role / Execution Stage | START / SUPPORT / CHECK / AVOID 等角色 |
| L4 | Constraint / Implementation Detail | 约束、输入输出格式、适用条件 |

输出：

- skill semantic ID。
- code assignments。
- code cards。
- code purity / usage entropy。

### `src/skillrq/decomposition`

负责 query 侧 latent skill decomposition。

核心对象：

- `query_encoder.py`: full query embedding。
- `path_selector.py`: 选择多个 semantic code paths。
- `residual_state.py`: 维护 uncovered skill need residual。
- `coverage.py`: 计算 step-wise coverage gain。

### `src/skillrq/hypergraph`

负责 skill co-occurrence hypergraph。

节点：

```text
skill
```

超边：

```text
一次任务共同需要的一组 skills
```

用途：

- training-time coverage supervision。
- inference-time implicit skill expansion。
- reranker 内部 graph prior。

### `src/skillrq/reranking`

负责候选 skill 的最终排序。

建议特征：

- Query-code match score。
- Dense relevance。
- Skill body evidence。
- Role compatibility。
- Hypergraph prior。
- Coverage gain。
- Redundancy penalty。
- Generic skill penalty。
- Constraint violation penalty。

### `src/skillrq/planning_support`

负责把推荐结果整理成给 LLM planner 使用的简洁 support。

默认输出：

```json
{
  "skill_id": "string",
  "name": "string",
  "rank": 1,
  "suggested_role": "START|SUPPORT|CHECK|FALLBACK",
  "matched_code_path": ["L1", "L2", "L3", "L4"],
  "code_support": "short text",
  "body_evidence": "short text",
  "implicit_tag": "optional short tag"
}
```

不默认暴露冗长 hypergraph explanation。

## 4. 数据流

```text
raw datasets
  -> data loaders
  -> canonical processed files
  -> embeddings
  -> baseline retrieval indexes
  -> SkillRQ codebook
  -> query code path decomposition
  -> explicit candidates
  -> hypergraph expansion
  -> role-aware reranking
  -> evaluation + reports
```

## 5. 命令设计

建议 CLI 命令：

```bash
python -m skillrq data build --config configs/data/skillret.yaml
python -m skillrq capability build --dataset api_bank
python -m skillrq capability build --dataset toolbench --skip-answer-trees --limit-tools 100 --limit-queries 50
python -m skillrq embed skills --config configs/experiments/baseline_skillret.yaml
python -m skillrq retrieve --config configs/experiments/baseline_skillret.yaml
python -m skillrq codebook build --config configs/experiments/rq_kmeans_skillret.yaml
python -m skillrq train --config configs/experiments/skillrq_no_hypergraph.yaml
python -m skillrq hypergraph build --config configs/experiments/skillrq_with_hypergraph.yaml
python -m skillrq evaluate --run runs/<run_id>
```

## 6. 实验输出规范

每次运行写入：

```text
runs/<timestamp>-<experiment_name>/
  config.yaml
  metrics.json
  predictions.jsonl
  code_paths.jsonl
  candidates.jsonl
  reranked.jsonl
  artifacts/
    codebook.json
    hypergraph.json
    code_cards.jsonl
```

报告导出：

```text
reports/
  tables/
    retrieval_metrics.csv
    coverage_metrics.csv
    code_quality_metrics.csv
    ablations.csv
  figures/
  cases/
    invoice_pdf_case.md
    implicit_skill_case.md
```

## 7. 配置约定

`configs/paths.yaml` 应集中声明数据位置：

```yaml
raw_root: /Users/sihan/code/skill-rec/data/raw
project_data_root: data
processed_root: data/processed
index_root: data/indexes
cache_root: data/cache
run_root: runs
report_root: reports
```

每个 experiment config 应记录：

- dataset。
- split。
- encoder。
- retrieval top_k。
- codebook levels 和 size。
- coverage settings。
- hypergraph settings。
- reranker feature weights。
- random seed。

## 8. 数据集到模块的映射

| 阶段 | 数据集 | 主要模块 | 不做什么 |
|---|---|---|---|
| Phase 1 | SkillRet | `data`、`encoding`、`retrieval`、`codebook`、`decomposition`、`reranking` | 不引入 Agent harness |
| Phase 2 | SkillRouter | `reranking`、`evidence`、`evaluation` | 不作为 SkillRQ 主体唯一训练集 |
| Phase 3 | SRA-Bench | `coverage`、`hypergraph`、`coverage_eval` | 单 gold skill 样本不作为 hyperedge 主体 |
| Phase 4 | SkillsBench | `agent_eval`、`planning_support` | 当前阶段不下载、不阻塞前三阶段 |

## 9. 最小可运行切片

为了尽快验证路线，第一版只需要实现这些文件：

```text
src/skillrq/data/loaders/skillret.py
src/skillrq/data/canonicalize.py
src/skillrq/encoding/text_builder.py
src/skillrq/retrieval/dense.py
src/skillrq/evaluation/metrics.py
scripts/build_processed_data.py
scripts/run_retrieval.py
configs/experiments/baseline_skillret.yaml
tests/test_data_loaders.py
tests/test_metrics.py
```

这个切片能先跑通：

```text
SkillRet raw data -> canonical files -> dense retrieval -> Recall@K/MRR@K
```

随后再叠加 SkillRQ codebook、coverage 和 hypergraph。这样路线稳，复杂度也不会一下子把项目压扁。
