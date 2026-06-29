# SkillRQ Processed Data Report

本文档说明 `/Users/sihan/code/SkillRQ/data/processed` 下当前已生成的 JSONL 文件字段，并统计 `queries.jsonl` 中每条 query 包含的 gold skills 数量。

## 文件概览

| 文件 | 行数 | 说明 |
|---|---:|---|
| `skills.jsonl` | 16,783 | 规范化后的 skill library |
| `queries.jsonl` | 68,256 | 规范化后的 query 数据，每条 query 包含 gold skill set |
| `qrels.jsonl` | 135,537 | query-skill relevance 标注 |
| `task_skill_sets.jsonl` | 68,256 | 每个 task/query 对应的 gold skill set |
| `roles.jsonl` | 135,537 | query-skill 角色占位标注，当前角色尚未分配 |
| `splits/train.jsonl` | 56,934 | train query split |
| `splits/dev.jsonl` | 6,325 | dev query split |
| `splits/test.jsonl` | 4,997 | test query split |

## `skills.jsonl`

每一行表示一个 skill。

| 字段 | 含义 |
|---|---|
| `skill_id` | 规范化后的 skill ID，目前沿用 SkillRet 原始 ID |
| `source_skill_id` | 原始数据集中的 skill ID |
| `source_dataset` | 来源数据集，当前为 `skillret` |
| `source_split` | 该 skill 首次保留的来源 split，取值如 `train` / `test` |
| `name` | skill 名称 |
| `namespace` | skill 命名空间 |
| `description` | skill 简短描述 |
| `body` | skill 正文内容 |
| `skill_md` | 原始 Skill Markdown 内容 |
| `domain_label` | 原始 domain 标签 |
| `operation_label` | 初始 operation 标签，目前来自 `primary_action` |
| `major` | 原始一级分类 |
| `sub` | 原始二级分类 |
| `primary_action` | skill 的主要动作 |
| `primary_object` | skill 的主要操作对象 |
| `author` | 作者 |
| `license` | license |
| `repo` | 来源仓库 |
| `source_url` | skill 来源 URL |
| `raw_url` | skill 原始文件 URL |
| `stars` | 仓库 stars |
| `installs` | 安装量 |

## `queries.jsonl`

每一行表示一个 query。

| 字段 | 含义 |
|---|---|
| `query_id` | 规范化后的 query ID，目前沿用 SkillRet 原始 ID |
| `source_query_id` | 原始数据集中的 query ID |
| `source_dataset` | 来源数据集，当前为 `skillret` |
| `source_split` | 原始 split，取值为 `train` / `test` |
| `query` | query 文本 |
| `original_query` | 原始 query 文本；若原始数据没有该字段则为 `null` |
| `gold_skill_ids` | gold skill ID 列表 |
| `gold_skill_names` | gold skill 名称列表 |
| `k` | 原始样本中的 gold skill 数量提示 |
| `generator_model` | 生成该 query 的模型名 |
| `difficulty` | 难度，占位字段，当前为 `null` |
| `domain` | query 领域，占位字段，当前为 `null` |

### Gold Skill 数量统计

统计字段：`gold_skill_ids.length`

| 指标 | 数值 |
|---|---:|
| query 总数 | 68,256 |
| 最小 gold skill 数量 | 1 |
| 最大 gold skill 数量 | 3 |
| 平均 gold skill 数量 | 1.985715541491 |

数量分布：

| gold skill 数量 | query 数 |
|---:|---:|
| 1 | 23,095 |
| 2 | 23,041 |
| 3 | 22,120 |

## `qrels.jsonl`

每一行表示一个 query 与一个 skill 的 relevance 标注。

| 字段 | 含义 |
|---|---|
| `query_id` | query ID |
| `skill_id` | skill ID |
| `relevance` | 相关性分数，当前正样本为 `1` |
| `source_dataset` | 来源数据集，当前为 `skillret` |
| `source_split` | 原始 split，取值为 `train` / `test` |

## `task_skill_sets.jsonl`

每一行表示一个 task/query 级别的 gold skill set。

| 字段 | 含义 |
|---|---|
| `task_id` | task ID，当前与 `query_id` 相同 |
| `query_id` | query ID |
| `source_dataset` | 来源数据集，当前为 `skillret` |
| `source_split` | 原始 split，取值为 `train` / `test` |
| `gold_skill_ids` | 该 task/query 对应的 gold skill ID 列表 |
| `num_gold_skills` | `gold_skill_ids` 的长度 |

## `roles.jsonl`

每一行表示一个 query-skill 对的角色标注占位。

| 字段 | 含义 |
|---|---|
| `query_id` | query ID |
| `skill_id` | skill ID |
| `role` | 角色标签，当前尚未分配，值为 `null` |
| `role_source` | 角色来源，当前为 `unassigned` |
| `source_dataset` | 来源数据集，当前为 `skillret` |
| `source_split` | 原始 split，取值为 `train` / `test` |

## `splits/*.jsonl`

`splits/train.jsonl`、`splits/dev.jsonl`、`splits/test.jsonl` 使用相同 schema。

| 字段 | 含义 |
|---|---|
| `query_id` | query ID |
| `split` | 规范化后的 split，取值为 `train` / `dev` / `test` |
| `source_dataset` | 来源数据集，当前为 `skillret` |
| `source_split` | 原始 split，取值为 `train` / `test` |

## 一致性检查

当前 `data/processed/stats.json` 中记录：

| 检查项 | 数值 |
|---|---:|
| `queries_without_gold_skills` | 0 |
| `missing_qrel_query_ids` | 0 |
| `missing_qrel_skill_ids` | 0 |

