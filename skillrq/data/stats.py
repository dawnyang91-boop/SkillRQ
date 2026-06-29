"""Dataset statistics for canonical SkillRQ files."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


def build_stats(
    skills: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    qrels: Sequence[Mapping[str, Any]],
    split_counts: Mapping[str, int],
) -> Mapping[str, Any]:
    gold_counts = [len(query["gold_skill_ids"]) for query in queries]
    multi_skill_queries = sum(1 for count in gold_counts if count > 1)
    category_counts = Counter(str(skill.get("major") or "unknown") for skill in skills)
    skill_ids = {skill["skill_id"] for skill in skills}
    query_ids = {query["query_id"] for query in queries}
    missing_qrel_skill_ids = sorted({qrel["skill_id"] for qrel in qrels if qrel["skill_id"] not in skill_ids})
    missing_qrel_query_ids = sorted({qrel["query_id"] for qrel in qrels if qrel["query_id"] not in query_ids})

    return {
        "source_dataset": "skillret",
        "skills": len(skills),
        "queries": len(queries),
        "qrels": len(qrels),
        "multi_skill_queries": multi_skill_queries,
        "single_skill_queries": len(queries) - multi_skill_queries,
        "queries_without_gold_skills": sum(1 for count in gold_counts if count == 0),
        "missing_qrel_skill_ids": len(missing_qrel_skill_ids),
        "missing_qrel_query_ids": len(missing_qrel_query_ids),
        "max_gold_skills_per_query": max(gold_counts) if gold_counts else 0,
        "avg_gold_skills_per_query": (sum(gold_counts) / len(gold_counts)) if gold_counts else 0.0,
        "splits": dict(split_counts),
        "top_major_categories": dict(category_counts.most_common(20)),
    }
