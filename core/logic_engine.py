from __future__ import annotations

from collections.abc import Iterable
from uuid import uuid4

import pandas as pd


def _round(value: float, places: int = 4) -> float:
    return round(float(value), places)


def _json_safe(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include="number")


def _statistical_summary(df: pd.DataFrame) -> dict:
    numeric = _numeric_frame(df)
    missing = {
        column: _round(percent, 2)
        for column, percent in (df.isna().mean() * 100).items()
        if percent > 0
    }
    means = {
        column: _round(value)
        for column, value in numeric.mean(numeric_only=True).dropna().items()
    }
    outliers = {}
    for column in numeric.columns:
        series = numeric[column].dropna()
        if len(series) < 3:
            continue
        std = series.std()
        if not std:
            continue
        z_scores = ((series - series.mean()) / std).abs()
        max_z = float(z_scores.max())
        if max_z >= 1.25:
            outliers[column] = {
                "max_abs_z_score": _round(max_z),
                "outlier_count": int((z_scores >= 1.25).sum()),
            }

    rolling = {}
    for column in numeric.columns:
        series = numeric[column].dropna()
        if len(series) >= 3:
            rolling[column] = [_round(value) for value in series.rolling(window=3).mean().dropna().tail(3)]

    correlations = {}
    if len(numeric.columns) >= 2 and len(numeric.dropna(how="all")) >= 2:
        corr = numeric.corr(numeric_only=True)
        for left_index, left in enumerate(corr.columns):
            for right in corr.columns[left_index + 1 :]:
                value = corr.loc[left, right]
                if pd.notna(value):
                    correlations[f"{left}__{right}"] = _round(value)

    return {
        "missing_percentage": missing,
        "numeric_means": means,
        "outlier_signals": outliers,
        "rolling_averages": rolling,
        "correlations": correlations,
    }


def _trigger_reason(summary: dict, depth: int, row_count: int, parent_summary: dict | None) -> str:
    reasons: list[str] = []
    if depth == 0:
        reasons.append("root analysis")
    if any(value >= 20 for value in summary["missing_percentage"].values()):
        reasons.append("high missingness")
    if summary["outlier_signals"]:
        reasons.append("outlier signal")
    strong_corr = [k for k, v in summary["correlations"].items() if abs(float(v)) >= 0.65]
    if strong_corr:
        reasons.append("strong correlation")
    if parent_summary:
        parent_means = parent_summary.get("numeric_means", {})
        for column, value in summary["numeric_means"].items():
            parent_value = parent_means.get(column)
            if parent_value not in (None, 0):
                delta = abs(float(value) - float(parent_value)) / abs(float(parent_value))
                if delta >= 0.25:
                    reasons.append("branch mean shift")
                    break
    if depth > 0 and row_count >= 2 and not reasons:
        reasons.append("stable branch segment")
    return "; ".join(dict.fromkeys(reasons))


def _confidence(summary: dict, trigger_reason: str, row_count: int) -> float:
    score = 0.25
    if row_count >= 5:
        score += 0.1
    if "root analysis" in trigger_reason:
        score += 0.1
    if summary["outlier_signals"]:
        score += 0.15
    if summary["correlations"]:
        score += 0.1
    if "branch mean shift" in trigger_reason:
        score += 0.15
    if summary["missing_percentage"]:
        score -= 0.05
    return max(0.05, min(0.95, round(score, 2)))


def _choose_split(df: pd.DataFrame, min_rows: int) -> tuple[str, str] | None:
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            continue
        counts = df[column].dropna().value_counts()
        if 1 < len(counts) <= 8 and any(count >= min_rows for count in counts):
            return column, "categorical"

    numeric = _numeric_frame(df)
    candidates = {
        column: numeric[column].dropna().var()
        for column in numeric.columns
        if numeric[column].dropna().nunique() > 1
    }
    if not candidates:
        return None
    return max(candidates, key=lambda column: candidates[column]), "numeric"


def _branches(df: pd.DataFrame, column: str, kind: str, min_rows: int) -> Iterable[tuple[str, pd.DataFrame]]:
    if kind == "categorical":
        for value, branch in df.groupby(column, dropna=True):
            if len(branch) >= min_rows:
                yield f"{column}={_json_safe(value)}", branch
        return

    median = df[column].median()
    lower = df[df[column] <= median]
    upper = df[df[column] > median]
    if len(lower) >= min_rows:
        yield f"{column}<={_round(median)}", lower
    if len(upper) >= min_rows:
        yield f"{column}>{_round(median)}", upper


def _build_node(
    df: pd.DataFrame,
    *,
    parent_id: str | None,
    depth: int,
    branch_label: str,
    max_depth: int,
    min_rows: int,
    parent_summary: dict | None,
) -> dict:
    node_id = str(uuid4())
    summary = _statistical_summary(df)
    trigger_reason = _trigger_reason(summary, depth, len(df), parent_summary)
    node = {
        "node_id": node_id,
        "parent_id": parent_id,
        "depth": depth,
        "branch_label": branch_label,
        "row_count": int(len(df)),
        "statistical_summary": summary,
        "trigger_reason": trigger_reason,
        "llm_prompt": "",
        "llm_response": "",
        "confidence_score": _confidence(summary, trigger_reason, len(df)),
        "children": [],
    }

    if depth >= max_depth or len(df) < min_rows * 2:
        return node

    split = _choose_split(df, min_rows)
    if not split:
        return node
    column, kind = split
    for label, branch in _branches(df, column, kind, min_rows):
        node["children"].append(
            _build_node(
                branch,
                parent_id=node_id,
                depth=depth + 1,
                branch_label=label,
                max_depth=max_depth,
                min_rows=min_rows,
                parent_summary=summary,
            )
        )
    return node


def build_analysis_tree(
    data: list[dict],
    *,
    user_goal: str,
    focus_columns: list[str] | None = None,
    max_depth: int = 2,
    min_rows: int = 2,
) -> dict:
    if not data:
        raise ValueError("data must contain at least one row")
    df = pd.DataFrame(data)
    if focus_columns:
        available = [column for column in focus_columns if column in df.columns]
        if available:
            df = df[available]
    max_depth = max(0, min(int(max_depth), 5))
    min_rows = max(1, int(min_rows))
    return _build_node(
        df,
        parent_id=None,
        depth=0,
        branch_label="root",
        max_depth=max_depth,
        min_rows=min_rows,
        parent_summary=None,
    )


def iter_nodes(root_node: dict) -> Iterable[dict]:
    yield root_node
    for child in root_node.get("children", []):
        yield from iter_nodes(child)


def synthesize_prediction(root_node: dict, user_goal: str) -> tuple[str, float]:
    nodes = list(iter_nodes(root_node))
    triggered = [node for node in nodes if node.get("trigger_reason")]
    strongest = sorted(triggered, key=lambda item: item.get("confidence_score", 0), reverse=True)[:3]
    reasons = [node["trigger_reason"] for node in strongest if node.get("trigger_reason")]
    average = sum(float(node.get("confidence_score", 0)) for node in triggered) / max(1, len(triggered))
    confidence = round(average, 2)
    if reasons:
        prediction = f"{user_goal}: strongest signals are " + "; ".join(reasons) + "."
    else:
        prediction = f"{user_goal}: no strong predictive branch emerged from the provided rows."
    return prediction, confidence
