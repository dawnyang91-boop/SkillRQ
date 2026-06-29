from skillrq.retrieval.bm25 import BM25Index
from skillrq.retrieval.dense import HashingDenseIndex
from skillrq.retrieval.metrics import evaluate_predictions
from skillrq.retrieval.types import Candidate, Query


def test_bm25_retrieves_matching_candidate():
    candidates = [
        Candidate("tool/weather", "weather forecast temperature city", "toy"),
        Candidate("tool/calendar", "calendar meeting agenda invite", "toy"),
    ]

    ranked = BM25Index(candidates).search("city weather forecast", top_k=1)

    assert ranked[0][0] == "tool/weather"


def test_hashing_dense_retrieves_matching_candidate():
    candidates = [
        Candidate("skill/python", "python pandas csv dataframe", "toy"),
        Candidate("skill/bgp", "bgp route leak oscillation network", "toy"),
    ]

    ranked = HashingDenseIndex(candidates, dimensions=128).search("detect bgp oscillation", top_k=1)

    assert ranked[0][0] == "skill/bgp"


def test_metrics_include_set_and_sequence_metrics():
    queries = [
        Query(
            query_id="q1",
            text="",
            gold_ids=["a", "b"],
            source_dataset="toy",
            sequence_ids=["a", "b"],
        )
    ]
    metrics = evaluate_predictions(queries, {"q1": ["a", "b"]}, [1, 2], task_type="tool")

    assert metrics["recall@1"] == 0.5
    assert metrics["completeness@2"] == 1.0
    assert metrics["tool_set_recall@2"] == 1.0
    assert metrics["first_tool_accuracy"] == 1.0
    assert metrics["transition_accuracy"] == 1.0
    assert metrics["kendall_tau"] == 1.0
