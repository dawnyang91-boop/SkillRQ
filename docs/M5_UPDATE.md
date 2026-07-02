# M5 Code Path Planner 问题诊断与修复计划书

## 0. 总目标

当前 M5 的目标不是放弃 codebook / code prediction，而是修复：

```text
code path prediction
→ code path bucket retrieval
→ candidate ranking
→ candidate recall
```

这条链路中的断裂问题。

本轮重构必须遵守以下原则：

1. 保留 codebook-centric 主线。
2. 保留 M4 soft multi-path code prediction。
3. 保留 M5 residual code path planning。
4. 不把系统改成普通 dense retrieval / 普通 cross-encoder retrieval。
5. 修复重点放在 code path 到 candidate 的可靠转换，以及 M5 对 M4 evidence 的复用。

---

# 1. 当前问题总览

## 问题 1：顶层 prediction 文件没有 candidate 字段，容易误判为 schema 不匹配

### 现象

诊断脚本显示：

```text
candidate fields: Counter()
avg candidates: None
rows_with_pred: 0
```

但这只是因为脚本只检查了顶层字段。

新版 M5 code-plan prediction 的候选不是放在：

```text
row["retrieved_capabilities"]
```

而是放在：

```text
row["residual_code_paths"][i]["retrieved_capabilities"]
row["code_plan"][i]["retrieved_capabilities"]
```

### 结论

这不是主因。`m5 evaluate` 本身读取的是 nested candidates。

### 解决方案

新增一个 nested prediction diagnostic 脚本，避免后续误判。

### 需要实现

新增文件：

```text
scripts/diagnose_m5_nested_candidates.py
```

功能：

1. 统计 prediction rows 数量。
2. 统计每条 query 的 residual steps 数量。
3. 统计 nested candidates 总数。
4. 统计每条 query 的平均 candidates 数量。
5. 统计 nested predicted candidate ids 与 gold ids 的全局交集。
6. 统计每个 step 的 candidate 数量分布。

### 验收标准

运行后必须输出：

```text
rows
total_steps
avg_steps
total_nested_candidates
avg_candidates_per_query
rows_with_candidates
unique_gold_ids
unique_pred_ids
global_id_intersection
step_candidate_counts
```

如果 `avg_candidates_per_query > 0`，说明 schema 不是主要问题。

---

# 2. 问题 2：M5 预测出的 code path 大多有效，但最终 candidate recall 接近 0

## 现象

当前检测结果：

```text
unique candidate code paths: 52483
unique predicted code paths: 20639
predicted path occurrences: 94837
path occurrence hit rate: 0.9058
```

说明 M5 预测出的 code path 有 90% 以上能在 candidate code index 里找到 bucket。

但是最终评估结果是：

```text
recall@100 = 0.000166
completeness@100 = 0
step_0_coverage_gain = 0.000166
step_1~step_5_coverage_gain = 0
```

## 判断

M5 并不是完全不会预测 code path。真正问题是：

```text
predicted code path exists
→ candidate bucket exists
→ 但 bucket 内取出的 candidate 基本不是 gold candidate
```

这说明问题集中在：

```text
code path bucket retrieval / bucket ranking
```

## 解决方案

新增 code path 到 candidate 的链路诊断。

### 需要实现

新增文件：

```text
scripts/diagnose_m5_codepath_to_candidate.py
```

需要统计：

1. `path_bucket_hit_rate`

   * predicted path 是否存在 candidate bucket。

2. `gold_path_covered_rate`

   * M5 predicted paths 是否覆盖 gold capability 的 native code path。

3. `gold_candidate_in_exact_bucket_rate`

   * gold candidate 是否在其 exact code path bucket 中。

4. `gold_candidate_rank_in_exact_bucket`

   * gold candidate 在 exact bucket 中的排名分布。

5. `rank<=20 / rank<=100`

   * 如果 gold candidate 在 exact bucket 中，按当前 bucket 原始顺序是否能进 top 20 / top 100。

6. `large_bucket_ratio`

   * M5 选中的 path 有多少落入 bucket size > 50 或 > 100 的大桶。

7. `generic_path_frequency`

   * 高频泛化路径，例如 `get_all`、`search_for`、`method_unknown_schema_light` 的出现频率。

### 验收标准

必须能判断当前问题属于以下哪一类：

```text
A. gold code path 没有被预测到
B. gold code path 被预测到了，但 bucket 内 gold candidate 排名靠后
C. bucket 太大，泛化 code path 霸榜
D. candidate id / gold id 存在映射问题
E. evaluator 读取正常，但 candidate ranking 失败
```

---

# 3. 问题 3：当前 `_retrieve_for_path()` 只靠 code overlap，bucket 内排序几乎无效

## 现象

当前逻辑类似：

```python
overlap = number of matched code levels
if overlap == 0:
    skip
score = path_probability * overlap_ratio
```

这会导致几个问题：

1. 只匹配 1 层 code 的 candidate 也能进入候选池。
2. 同一 exact code path bucket 内所有 candidate 分数几乎一样。
3. 没有利用 query text。
4. 没有利用 candidate schema / description。
5. 没有利用 M4 已经召回的 candidate evidence。
6. 大 bucket 会把很多无关 candidate 排到前面。

## 解决方案

将 `_retrieve_for_path()` 改为 exact-first bucket retrieval。

### 需要修改文件

```text
skillrq/m5/planning.py
```

重点修改函数：

```text
_retrieve_for_path()
```

### 新逻辑

候选召回按优先级分层：

```text
Level 1: exact code path match, overlap = 4
Level 2: prefix match L1+L2+L3, overlap >= 3
Level 3: prefix match L1+L2, overlap >= 2
Level 4: fallback overlap >= 2，限制数量
```

不要让 `overlap = 1` 的 candidate 直接进入前排。

### 排序分数建议

```text
score =
  2.0 * exact_match
+ 0.8 * prefix_match_ratio
+ 0.5 * matched_level_ratio
+ 0.5 * m4_candidate_prior
+ 0.2 * path_probability
- 0.05 * log(1 + bucket_size)
```

其中：

```text
exact_match = 1 if overlap == 4 else 0
prefix_match_ratio = prefix_matched_levels / 4
matched_level_ratio = overlap / 4
m4_candidate_prior = 1 if candidate_id in M4 retrieved candidates else 0
bucket_size = number of candidates under the exact predicted code path
```

### 验收标准

修改后重新运行 M5 predict + evaluate。

对比旧版指标：

```text
old recall@100 = 0.000166
old completeness@100 = 0
```

新版至少应满足：

```text
recall@100 明显大于旧版
step_0_coverage_gain 明显上升
candidate_redundancy_ratio 不显著恶化
```

如果 recall 仍接近 0，说明问题不只在 `_retrieve_for_path()`，需要继续查 gold path covered rate 和 M4 evidence reuse。

---

# 4. 问题 4：M5 丢掉了 M4 retrieved candidates

## 现象

M4 soft-multipath predict 已经输出：

```text
predicted_code_paths
retrieved_capabilities
```

但 M5 当前 `_load_m4_predictions()` 只读取：

```text
predicted_code_paths
```

导致 M5 完全丢弃了 M4 的 candidate evidence。

## 判断

这会造成：

```text
M4 已经召回 gold candidate
但 M5 重新按 code overlap 扫库后把 gold candidate 丢掉
```

这是当前 M5 recall 接近 0 的高风险原因。

## 解决方案

修改 `_load_m4_predictions()`，让 M5 同时继承：

```text
M4 predicted_code_paths
M4 retrieved_capabilities
```

### 需要修改文件

```text
skillrq/m5/planning.py
```

### 修改前

```python
def _load_m4_predictions(path):
    if not path:
        return {}
    return {
        str(row["query_id"]): list(row.get("predicted_code_paths") or [])
        for row in read_jsonl(path)
    }
```

### 修改后

```python
def _load_m4_predictions(path):
    if not path:
        return {}
    return {
        str(row["query_id"]): {
            "predicted_code_paths": list(row.get("predicted_code_paths") or []),
            "retrieved_capabilities": list(row.get("retrieved_capabilities") or []),
        }
        for row in read_jsonl(path)
    }
```

同步修改调用处：

```python
m4_item = predictions.get(str(query["query_id"])) or {}
predicted_paths = m4_item.get("predicted_code_paths") or _oracle_predicted_paths(query)
m4_retrieved = m4_item.get("retrieved_capabilities") or []
```

然后将 `m4_retrieved` 传入 `_retrieve_for_path()`。

### 新增指标

在 M5 prediction summary 中加入：

```text
m4_candidate_reuse_rate
m4_hit_m5_miss_count
m4_hit_m5_miss_rate
m4_miss_m5_hit_count
m4_miss_m5_hit_rate
```

### 验收标准

运行对照实验：

```text
A. M5 不使用 M4 retrieved_capabilities
B. M5 使用 M4 retrieved_capabilities prior
```

若 B 的 recall@100 明显高于 A，则说明 M4 evidence reuse 有效。

---

# 5. 问题 5：训练阶段强行把 gold path 塞进 pool，导致训练和推理不一致

## 现象

M5 planning data 构造时，如果 M4 predicted paths 没有 gold path，代码会把 gold path 强行加入 pool。

这会导致训练目标变成：

```text
即使 M4 没预测到 gold path，也要求 M5 预测 gold path
```

这对当前轻量 planner 来说太难，也会导致模型记忆高频 path，而不是学习 realistic planning。

## 解决方案

将 M5 planning data 分成两个模式：

```text
oracle mode
realistic mode
```

### Oracle mode

保留当前逻辑：

```text
如果 gold path 不在 M4 predicted paths 中，也强行加入 pool
```

用途：

```text
只用于上界实验
```

### Realistic mode

默认模式：

```text
只允许 M5 在 M4 predicted paths 或可控 expansion paths 中选择
不强行加入 gold path
```

用途：

```text
用于真实训练和真实推理
```

### 需要修改文件

```text
skillrq/m5/planning.py
skillrq/cli.py
```

### CLI 参数建议

在 `m5 prepare --model-kind code-plan` 中新增：

```bash
--oracle-expand-gold-paths
```

默认关闭。

### 数据统计新增字段

`stats.json` 中增加：

```text
gold_path_in_m4_pool_rate
gold_query_fully_covered_by_m4_paths_rate
gold_path_missing_from_m4_pool_count
oracle_expanded_gold_path_count
```

### 验收标准

运行：

```bash
python3 -m skillrq m5 prepare \
  --target capability \
  --model-kind code-plan \
  --m4-prediction-path <m4_predictions> \
  --output-root data/processed/m5_code_plan_realistic/capability
```

和：

```bash
python3 -m skillrq m5 prepare \
  --target capability \
  --model-kind code-plan \
  --m4-prediction-path <m4_predictions> \
  --oracle-expand-gold-paths \
  --output-root data/processed/m5_code_plan_oracle/capability
```

分别训练和评估，比较：

```text
realistic M5
oracle M5
M4 baseline
```

---

# 6. 问题 6：M5 现在更像 path generator，而不是 path planner/ranker

## 现象

当前 `_predict_next_step()` 会分别预测 L1/L2/L3/L4，然后 beam 组合出 code path。

问题：

1. 容易组合出高频泛化 path。
2. 与 M4 predicted path distribution 的联系不够紧。
3. M5 本应在 M4 path pool 上做 residual planning，而不是重新生成任意 path。
4. 容易过拟合 code label。

## 解决方案

中期将 M5 从 “code path generator” 改成 “code path reranker/planner”。

### 新 M5 输入

```text
query
current residual state
M4 predicted code path pool
selected code paths
covered roles
covered operations
covered schema constraints
```

### 新 M5 输出

```text
score(q, state, candidate_code_path)
stop_probability
expected_coverage_gain
role suitability
```

### 训练方式

对每一步构造：

```text
positive path: 能覆盖尚未覆盖 gold path / gold candidate 的 path
negative path: M4 pool 中不能带来 coverage gain 的 path
```

训练 pairwise ranking：

```text
score(positive_path) > score(negative_path)
```

### 损失函数建议

```text
L =
  L_pairwise_path_rank
+ λ_gain * L_gain_cls
+ λ_stop * L_stop_balanced_bce
+ λ_role * L_role_aux
```

不要继续让四层 code CE 主导 M5。

### 验收标准

新 M5 的 dev 指标不再只看：

```text
dev/path_exact_match
dev/code_loss
```

而是新增：

```text
dev/gold_path_covered_rate
dev/path_pool_recall@K
dev/candidate_recall@K
dev/avg_selected_steps
```

---

# 7. 问题 7：coverage_loss 数值太小，无法真正优化 coverage

## 现象

训练曲线中：

```text
code_loss 远大于 coverage_loss
coverage_loss 虽然下降，但 final recall 几乎不变
```

说明当前 coverage_loss 没有直接优化最终 coverage。

当前 coverage_loss 是：

```text
MSE(sigmoid(coverage_gain), expected_coverage_gain)
```

而 `expected_coverage_gain` 又是 role / operation / schema / gold_path 的抽象 gain，不是 candidate-level recall gain。

## 解决方案

将 coverage 从 MSE regression 改为 ranking/classification。

### 短期做法

新增：

```text
gain_positive_label = 1 if selected path 能覆盖新 gold path / gold candidate else 0
```

使用：

```text
BCEWithLogitsLoss
```

### 中期做法

构造 positive / negative code path pair：

```text
positive = coverage_gain > 0
negative = coverage_gain = 0
```

使用：

```text
MarginRankingLoss
```

### 推荐损失

```text
L =
  1.0 * L_path_rank
+ 2.0 * L_gain_bce
+ 0.5 * L_stop
+ 0.2 * L_role
```

### 验收标准

加入 coverage ranking 后，观察：

```text
dev/gold_path_covered_rate
dev/candidate_recall@100
step_1~step_5_coverage_gain
```

如果 step_1 之后仍然是 0，说明 planner 没有学会 residual coverage。

---

# 8. 问题 8：stop head 可能导致过早停止或评估无效

## 现象

当前预测结果中没有显式 `stop_step` 统计。虽然 prediction 平均 path 数量约 9.48，但最终 step coverage gain 只有 step 0 有极小值，step 1 以后全为 0。

## 解决方案

新增 stop 行为诊断，并做 stop ablation。

### 需要实现

在 M5 predict summary 中加入：

```text
avg_steps
stop_probability_mean
stop_probability_by_step
stopped_query_count
max_step_query_count
```

### 需要跑的对照实验

```text
A. 原始 stop_threshold = 0.55
B. stop_threshold = 0.70
C. stop_threshold = 0.90
D. disable stop, force max_steps = 6
E. min_steps = 2
F. min_steps = 3
```

### 验收标准

如果 D/E/F 显著提高 recall，说明 stop head 或 selected path update 逻辑有问题。

如果 D/E/F 仍然无效，说明主因是 path selection / candidate ranking。

---

# 9. 问题 9：dev loss 与最终 recall 不一致

## 现象

训练曲线显示：

```text
train loss 持续下降
dev code loss 先下降后上升
dev path_exact_match 上升
但 final recall 接近 0
```

说明当前训练过程监控的 dev 指标不能代表最终任务质量。

## 解决方案

训练时增加 retrieval-level dev evaluation。

### 新增 dev 指标

每个 epoch 记录：

```text
dev/gold_path_covered_rate
dev/candidate_recall@20
dev/candidate_recall@100
dev/completeness@100
dev/avg_selected_steps
dev/exact_bucket_hit_rate
dev/m4_candidate_reuse_rate
```

### Early stopping 指标

不要使用：

```text
dev/loss
dev/code_loss
```

作为唯一 early stopping 标准。

应改为：

```text
dev/candidate_recall@100
```

或：

```text
dev/gold_path_covered_rate + dev/candidate_recall@100
```

### 验收标准

保存：

```text
best_by_dev_loss.pt
best_by_dev_recall.pt
last.pt
```

分别 predict + evaluate，比较差异。

---

# 10. 执行顺序

## Phase 0：只做诊断，不改模型

### 任务 0.1：新增 nested candidate 诊断

文件：

```text
scripts/diagnose_m5_nested_candidates.py
```

输出：

```text
avg_nested_candidates
global_id_intersection
query_hit_rate
step_hit_distribution
```

### 任务 0.2：新增 code path 到 candidate 诊断

文件：

```text
scripts/diagnose_m5_codepath_to_candidate.py
```

输出：

```text
path_bucket_hit_rate
gold_path_covered_rate
gold_candidate_rank_in_exact_bucket
large_bucket_ratio
```

### 任务 0.3：新增 M4 → M5 candidate evidence 诊断

文件：

```text
scripts/diagnose_m4_m5_candidate_reuse.py
```

输出：

```text
m4_hit_rate
m5_hit_rate
m4_hit_m5_miss
m4_miss_m5_hit
```

### Phase 0 验收

必须明确回答：

```text
M5 recall 低是因为：
A. gold path 没预测到
B. gold path 预测到了但 bucket ranking 失败
C. M4 本来召回了 gold，但 M5 丢掉了
D. ID mapping 错误
E. evaluator schema 错误
```

### Phase 0 实现状态

已新增以下脚本：

```text
scripts/diagnose_m5_nested_candidates.py
scripts/diagnose_m5_codepath_to_candidate.py
scripts/diagnose_m4_m5_candidate_reuse.py
```

推荐执行：

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

本地旧版 `runs/m5_residual_selector/predictions/capability/predictions.jsonl` smoke test 结果显示：nested candidates 能被正常读取，`avg_candidates_per_query` 大于 0，说明 evaluator schema 不是主要问题；`gold_path_covered_rate` 很低，下一阶段应优先检查 path selection 与 M4 evidence reuse。

---

## Phase 1：修 M5 retrieval，不改训练

### 任务 1.1：改 `_retrieve_for_path()` 为 exact-first

文件：

```text
skillrq/m5/planning.py
```

### 任务 1.2：加入 bucket size penalty

抑制：

```text
get_all
search_for
method_unknown_schema_light
toolbench_answer_tree
```

这类泛化大桶。

### 任务 1.3：重新 predict + evaluate

命令模板：

```bash
python3 -m skillrq m5 predict \
  --target capability \
  --model-kind code-plan \
  --m4-data-root data/processed/m4_sequence_eval/capability \
  --m4-prediction-path <m4_predictions.jsonl> \
  --checkpoint-root runs/m5_code_path_planner/capability \
  --output-root runs/m5_code_path_planner/predictions/capability_sequence_eval_exact_first \
  --split test \
  --top-n-paths 16 \
  --candidates-per-step 50 \
  --stop-threshold 0.7 \
  --enable-exact-first-retrieval \
  --device cuda

python3 -m skillrq m5 evaluate \
  --prediction-path runs/m5_code_path_planner/predictions/capability_sequence_eval_exact_first/predictions.jsonl \
  --output-path reports/tables/m5_exact_first.json \
  --top-k 5,10,20,50,100 \
  --set-metric-name tool_set_recall
```

### Phase 1 验收

新版：

```text
recall@100 > old recall@100
step_0_coverage_gain > old step_0_coverage_gain
```

### Phase 1 实现状态

已修改：

```text
skillrq/m5/planning.py
skillrq/cli.py
tests/test_m5_code_plan.py
```

Phase 1 现在通过显式参数启用：

```bash
--enable-exact-first-retrieval
```

默认不启用，保持 Phase 2 M4-prior 行为不变。启用后 `_retrieve_for_path()` 使用 exact-first 排序：

```text
score =
  2.0 * exact_match
+ 0.8 * prefix_match_ratio
+ 0.5 * matched_level_ratio
+ 0.5 * m4_candidate_prior
+ 0.2 * path_probability
- 0.05 * log(1 + exact_bucket_size)
- generic_path_penalty
```

其中 `generic_path_penalty` 会抑制：

```text
get_all
search_for
method_unknown_schema_light
toolbench_answer_tree
```

为了做干净消融，新增：

```bash
--disable-m4-candidate-prior
```

推荐先跑两个版本：

```bash
# Phase 1 only: exact-first，不使用 M4 candidate prior
python3 -m skillrq m5 predict \
  --target capability \
  --model-kind code-plan \
  --m4-data-root data/processed/m4_sequence_eval/capability \
  --m4-prediction-path runs/m4_query_to_code/predictions/soft_multipath/capability_sequence_eval/predictions.jsonl \
  --checkpoint-root runs/m5_code_path_planner/capability_sequence_eval \
  --output-root runs/m5_code_path_planner/predictions/capability_sequence_eval_exact_first_only \
  --max-steps 6 \
  --top-n-paths 16 \
  --candidates-per-step 20 \
  --stop-threshold 0.55 \
  --split sequence_test \
  --enable-exact-first-retrieval \
  --disable-m4-candidate-prior \
  --device cuda

# Phase 1 + Phase 2: exact-first + M4 candidate prior
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

`predict_summary.json` 与 SwanLab 会记录：

```text
enable_exact_first_retrieval
use_m4_candidate_prior
```

---

## Phase 2：复用 M4 retrieved candidates

### 任务 2.1：修改 `_load_m4_predictions()`

保留：

```text
predicted_code_paths
retrieved_capabilities
```

### 任务 2.2：M5 retrieval 加 M4 candidate prior

候选排序加入：

```text
m4_candidate_prior
```

### 任务 2.3：新增 M4 reuse 统计

summary 输出：

```text
m4_candidate_reuse_rate
m4_hit_m5_miss_rate
```

### Phase 2 验收

对比：

```text
M5 exact-first without M4 prior
M5 exact-first with M4 prior
```

如果使用 M4 prior 后 recall@100 上升，保留该改动。

### Phase 2 实现状态

已修改：

```text
skillrq/m5/planning.py
tests/test_m5_code_plan.py
```

实现内容：

1. `_load_m4_predictions()` 现在同时读取 `predicted_code_paths` 与 `retrieved_capabilities`。
2. `predict_code_path_plan()` 会为每条 query 取出 M4 retrieved candidates，并传入 `_retrieve_for_path()`。
3. `_retrieve_for_path()` 新增 M4 candidate prior：

```text
score =
  original_code_overlap_score
+ 2.0 * m4_rank_prior
+ 0.5 * m4_code_match_score
+ 0.2 * selected_path_overlap_ratio
```

4. M5 predict 跨 residual steps 维护 `used_candidate_ids`，防止 M4 top candidates 在每一步重复出现。
5. M5 prediction rows 额外输出：

```text
m4_candidate_count
m4_candidate_reuse_count
residual_code_paths[].m4_candidate_reuse_count
retrieved_capabilities[].m4_candidate_prior
retrieved_capabilities[].m4_candidate_rank
retrieved_capabilities[].m4_code_match_score
retrieved_capabilities[].retrieval_source
```

6. `predict_summary.json` 与 SwanLab 额外记录：

```text
m4_candidate_reuse_rate
m4_hit_rate
m5_hit_rate
m4_hit_m5_miss_count
m4_hit_m5_miss_rate
m4_miss_m5_hit_count
m4_miss_m5_hit_rate
```

7. 为降低 M5 predict 的 CPU 开销，已新增 candidate retrieval index：

```text
exact_path -> candidates
L1/L2/L3 prefix -> candidates
L1/L2 prefix -> candidates
candidate_id -> candidate
```

`_retrieve_for_path()` 现在只访问 selected code path 对应的 exact / prefix buckets，并合并 M4 retrieved candidates，不再每个 step 遍历全量 candidates。`predict_summary.json` 会记录 `candidate_exact_path_buckets`、`candidate_l123_buckets` 和 `candidate_l12_buckets`，方便确认索引已生效。

推荐重新运行：

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
  --device cuda

python3 -m skillrq m5 evaluate \
  --prediction-path runs/m5_code_path_planner/predictions/capability_sequence_eval_m4_prior/predictions.jsonl \
  --output-path reports/tables/m5_code_plan_m4_prior_sequence_test.json \
  --top-k 5,10,20,50,100 \
  --set-metric-name tool_set_recall
```

然后复跑 Phase 0 的三个诊断脚本，对比 `m4_hit_m5_miss_rate` 是否从约 `0.9216` 显著下降，`m4_candidate_reuse_rate` 与 `m5_hit_rate` 是否明显上升。

---

## Phase 3：修 realistic / oracle 训练数据分离

### 任务 3.1：新增 CLI 参数

```bash
--oracle-expand-gold-paths
```

默认关闭。

### 任务 3.2：prepare 生成两套数据

```text
data/processed/m5_code_plan_realistic/capability
data/processed/m5_code_plan_oracle/capability
```

### 任务 3.3：分别训练和评估

比较：

```text
M5 realistic
M5 oracle
M4 baseline
```

### Phase 3 验收

必须输出：

```text
gold_path_in_m4_pool_rate
oracle_upper_bound_recall@100
realistic_recall@100
```

---

## Phase 4：修改 M5 训练目标

### 任务 4.1：降低 code CE 主导地位

短期可以先降低 code loss 权重。

### 任务 4.2：coverage 从 MSE 改为 BCE / pairwise ranking

新增：

```text
coverage_gain_label
positive_path
negative_path
```

### 任务 4.3：stop loss 使用 class-balanced BCE

防止 stop head 过早停止。

### Phase 4 验收

训练后必须检查：

```text
dev/gold_path_covered_rate
dev/candidate_recall@100
step_1~step_5_coverage_gain
```

---

## Phase 5：中期重构 M5 为 code path reranker/planner

### 目标

将 M5 从：

```text
生成 L1/L2/L3/L4 code path
```

改为：

```text
对 M4 predicted code path pool 进行 residual reranking/planning
```

### 新模型输入

```text
query
planner_state
candidate_code_path
candidate_code_path_verbalization
selected_paths
covered_roles
covered_operations
covered_schema_constraints
```

### 新模型输出

```text
path_score
coverage_gain_score
stop_score
role_score
```

### 新训练目标

```text
positive path score > negative path score
```

### Phase 5 验收

新 M5 至少要优于：

```text
M4 soft-multipath baseline
M5 exact-first heuristic
M5 old generator
```

---

# 11. 最终实验矩阵

至少跑以下实验：

```text
E0: M4 soft-multipath baseline
E1: Old M5 code-plan
E2: M5 exact-first retrieval
E3: M5 exact-first + bucket penalty
E4: M5 exact-first + M4 candidate prior
E5: M5 realistic training
E6: M5 oracle training upper bound
E7: M5 disable stop
E8: M5 min_steps=2 / min_steps=3
E9: M5 pairwise coverage ranking
```

每个实验报告：

```text
recall@5/10/20/50/100
completeness@5/10/20/50/100
tool_set_recall@5/10/20/50/100
candidate_redundancy_ratio
redundant_code_path_ratio
step_i_coverage_gain
gold_path_covered_rate
m4_candidate_reuse_rate
large_bucket_ratio
```

---

# 12. 最终判断标准

本轮修复成功的最低标准：

```text
M5 recall@100 明显高于当前 0.000166
M5 不再把 step_1~step_5 coverage gain 全部打成 0
M5 能证明 code path bucket retrieval 不是空转
M5 能复用 M4 candidate evidence
M5 realistic / oracle 的差距被明确量化
```

理想结果：

```text
M5 realistic > M4 soft-multipath baseline
M5 exact-first + M4 prior > old M5
M5 oracle upper bound 明显高于 realistic，证明 M4 path pool coverage 是后续优化方向
```

---

# 13. 本轮不建议优先做的事情

暂时不要优先做：

```text
1. 盲目加大 hidden_dim
2. 盲目增加 epoch
3. 只靠 dropout / weight decay 解决问题
4. 直接改成普通 dense retrieval
5. 直接上普通 cross-encoder 替代 codebook
6. 在没有 oracle upper bound 的情况下继续调 loss
```

因为当前主要问题不是模型容量，而是：

```text
code path → candidate retrieval/ranking
```

这条链路没有对齐最终 recall。
