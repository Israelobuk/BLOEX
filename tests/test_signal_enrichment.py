from core.context_parser import parse_structured_context
from core.signal_enrichment import enrich_explainer_result_with_signals


def test_parse_structured_context_extracts_rows_from_plain_text():
    context = """
    Entry A, score 91, 4 signals, outcome good.
    Entry B, score 62, 1 signal, outcome weak.
    Entry C, score 88, 3 signals, outcome good.
    """

    parsed = parse_structured_context(context)

    assert parsed is not None
    assert len(parsed["data"]) == 3
    assert "score" in parsed["focus_columns"]
    assert "signal" in parsed["focus_columns"]
    assert "outcome" in parsed["focus_columns"]
    assert parsed["data"][0]["outcome"] == "outcome good"
    assert parsed["data"][1]["outcome"] == "outcome weak"


def test_parse_structured_context_does_not_treat_row_text_as_csv_header():
    context = "Entry A, score 91, 4 signals, outcome good.\nEntry B, score 62, 1 signal, outcome weak."

    parsed = parse_structured_context(context)

    assert parsed is not None
    assert "score" in parsed["focus_columns"]
    assert "score_9" not in parsed["focus_columns"]


def test_parse_structured_context_returns_none_for_unstructured_text():
    parsed = parse_structured_context("Nvidia is a company that makes GPUs and AI chips.")

    assert parsed is None


def test_signal_enrichment_merges_recursive_read_into_existing_result():
    result = {
        "answer": "The model says the result is likely strong.",
        "black_box_explanation": "The model focused on the observed evidence.",
        "evidence_claims": [],
        "uncertainty": [],
        "confidence_reason": "Confidence is limited.",
    }
    signal_result = {
        "final_prediction": "The stronger result appears linked to higher scores and more signals.",
        "final_confidence": 0.62,
        "root_node": {
            "node_id": "root",
            "parent_id": None,
            "depth": 0,
            "branch_label": "root",
            "row_count": 3,
            "statistical_summary": {
                "numeric_means": {"score": 76.5, "signal": 2.5, "outcome_flag": 0.67},
                "missing_percentage": {},
                "outlier_signals": {},
                "rolling_averages": {},
                "correlations": {"signal__outcome_flag": 0.91},
            },
            "trigger_reason": "root analysis; strong correlation",
            "llm_prompt": "",
            "llm_response": "Recursive signal read says signal count tracks the stronger outcome.",
            "confidence_score": 0.62,
            "children": [],
        },
    }

    enriched = enrich_explainer_result_with_signals(result, signal_result)

    assert "Recursive signal read" in enriched["black_box_explanation"]
    assert enriched["signal_analysis"]["final_prediction"] == signal_result["final_prediction"]
    assert enriched["evidence_claims"]
    assert "structured signals" in enriched["confidence_reason"]
