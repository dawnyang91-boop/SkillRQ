# Capability Processed Data Report

Data directory: `/Users/sihan/code/SkillRQ/data/processed/capability`

## Files

| File | Rows | Fields | Description |
|---|---:|---|---|
| `capabilities.jsonl` | 64,645 | `api_name, api_schema, capability_id, capability_type, category, description, domain, endpoint, input_schema, method, name, optional_parameters, output_schema, parameters, provider, raw, required_parameters, source_capability_id, source_dataset, tool_name` | Capability objects: tools, APIs, functions, and runtime actions. |
| `capability_queries.jsonl` | 329,625 | `available_capability_ids, final_answer, gold_capability_ids, intermediate_observations, query, query_id, raw, source_dataset, source_query_id, source_split, success, tool_arguments, tool_call_sequence, tool_calls_per_trajectory, unique_tools_per_query` | User instructions, dialogues, tasks, and trajectory-only records. |
| `capability_qrels.jsonl` | 813,490 | `capability_id, query_id, relevance, source_dataset, source_split` | Query-capability relevance labels. |
| `capability_sequences.jsonl` | 447,066 | `arguments, capability_id, observation, query_id, source_dataset, source_split, step_index` | Step-wise tool/API calls extracted from trajectories. |

## Summary

| Metric | Value |
|---|---:|
| `capabilities` | 64,645 |
| `queries` | 329,625 |
| `qrels` | 813,490 |
| `sequences` | 447,066 |
| `min_unique_tools_per_query` | 0 |
| `max_unique_tools_per_query` | 10 |
| `avg_unique_tools_per_query` | 2.467926 |
| `min_tool_calls_per_trajectory` | 0 |
| `max_tool_calls_per_trajectory` | 5 |
| `avg_tool_calls_per_trajectory` | 1.356287 |

## Queries By Dataset

| Dataset | Queries |
|---|---:|
| `api_bank` | 263 |
| `toolbench` | 329,362 |

## Notes

- `unique_tools_per_query` is the size of the unique gold capability set and is used for recommendation / set selection.
- `tool_calls_per_trajectory` is the length of the extracted execution sequence and is used for execution-order analysis.
- ToolBench instruction records and API-Bank dialogues share the same canonical schema.
- ToolBench answer-tree records are retained as trajectory-only examples when full answer extraction is enabled.
