from skillrq.agent_sim.vllm_runner import _normalize_llm_plan, parse_tool_call_plan


def test_parse_and_normalize_vllm_tool_call_plan():
    text = """
    {
      "tool_calls": [
        {
          "step": 1,
          "tool_name": "search_hotels",
          "candidate_id": "tool_search",
          "arguments": {"city": "Paris"},
          "rationale": "needed for hotel search"
        }
      ],
      "final_answer_plan": "Search hotels first."
    }
    """
    record = {
        "query_id": "q1",
        "query": "Find a hotel.",
        "gold_ids": ["tool_search"],
        "sequence_ids": ["tool_search"],
        "capability_plan": [
            {
                "candidate_tools": [
                    {"candidate_id": "tool_search", "name": "search_hotels", "suggested_role": "START"}
                ]
            }
        ],
    }

    parsed = parse_tool_call_plan(text)
    normalized = _normalize_llm_plan(record, parsed, raw_text=text)

    assert parsed["parse_success"] is True
    assert normalized["tool_calls"][0]["candidate_id"] == "tool_search"
    assert normalized["tool_calls"][0]["grounding"] == "prompt_candidate"
    assert normalized["parse_success"] is True


def test_parse_invalid_vllm_output_returns_empty_plan():
    parsed = parse_tool_call_plan("not json")

    assert parsed["parse_success"] is False
    assert parsed["tool_calls"] == []
