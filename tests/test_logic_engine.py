from core.logic_engine import build_analysis_tree, synthesize_prediction


def test_build_analysis_tree_calculates_statistics_and_children():
    rows = [
        {"region": "north", "revenue": 100, "tickets": 2, "churned": 0},
        {"region": "north", "revenue": 120, "tickets": 3, "churned": 0},
        {"region": "south", "revenue": 400, "tickets": 9, "churned": 1},
        {"region": "south", "revenue": 900, "tickets": 15, "churned": 1},
        {"region": "west", "revenue": None, "tickets": 5, "churned": 0},
    ]

    root = build_analysis_tree(
        rows,
        user_goal="Find churn risk",
        focus_columns=["region", "revenue", "tickets", "churned"],
        max_depth=2,
        min_rows=2,
    )

    assert root["branch_label"] == "root"
    assert root["row_count"] == 5
    assert root["statistical_summary"]["missing_percentage"]["revenue"] == 20.0
    assert root["statistical_summary"]["numeric_means"]["tickets"] == 6.8
    assert "revenue" in root["statistical_summary"]["outlier_signals"]
    assert "revenue" in root["statistical_summary"]["rolling_averages"]
    assert "revenue__tickets" in root["statistical_summary"]["correlations"]
    assert root["trigger_reason"]
    assert root["confidence_score"] > 0
    assert root["children"]
    assert all(child["parent_id"] == root["node_id"] for child in root["children"])


def test_synthesize_prediction_uses_triggered_nodes_for_confidence():
    root = {
        "trigger_reason": "root analysis",
        "confidence_score": 0.4,
        "children": [
            {"trigger_reason": "outlier signal", "confidence_score": 0.8, "children": []},
            {"trigger_reason": "", "confidence_score": 0.2, "children": []},
        ],
    }

    prediction, confidence = synthesize_prediction(root, "Find churn risk")

    assert "Find churn risk" in prediction
    assert "outlier signal" in prediction
    assert confidence == 0.6
