from __future__ import annotations

import csv
import io
import json
import re
from typing import Any


def _clean_key(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if cleaned.endswith("s") and len(cleaned) > 3:
        cleaned = cleaned[:-1]
    return cleaned or "value"


def _coerce_value(value: str) -> Any:
    clean = value.strip().strip(".").strip()
    if not clean:
        return ""
    try:
        if "." in clean:
            return float(clean.replace(",", ""))
        return int(clean.replace(",", ""))
    except ValueError:
        return clean


def _focus_columns(rows: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return columns


def _enough_structure(rows: list[dict[str, Any]]) -> bool:
    if len(rows) < 2:
        return False
    shared_columns = set(rows[0])
    for row in rows[1:]:
        shared_columns &= set(row)
    numeric_cells = 0
    for row in rows:
        numeric_cells += sum(1 for value in row.values() if isinstance(value, (int, float)))
    return len(shared_columns) >= 2 and numeric_cells >= 2


def _parse_json_context(context: str) -> list[dict[str, Any]] | None:
    try:
        data = json.loads(context)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict):
        data = data.get("data") or data.get("rows")
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        return None
    return [dict(item) for item in data]


def _parse_csv_context(context: str) -> list[dict[str, Any]] | None:
    lines = [line for line in context.splitlines() if line.strip()]
    if len(lines) < 2 or "," not in lines[0]:
        return None
    header_cells = [cell.strip() for cell in lines[0].split(",")]
    if any(re.search(r"\d", cell) for cell in header_cells):
        return None
    try:
        reader = csv.DictReader(io.StringIO("\n".join(lines)))
        rows = [{_clean_key(k): _coerce_value(v) for k, v in row.items() if k} for row in reader]
    except csv.Error:
        return None
    return rows or None


def _parse_plaintext_context(context: str) -> list[dict[str, Any]] | None:
    rows: list[dict[str, Any]] = []
    for raw_line in context.splitlines():
        line = raw_line.strip().strip('"').strip()
        if not line or len(line.split()) < 4:
            continue
        row: dict[str, Any] = {}
        parts = [part.strip().strip(".") for part in re.split(r"[,;|]", line) if part.strip()]
        if len(parts) < 2:
            continue

        text_fields: list[str] = []
        for part in parts:
            matched = False

            # Handles "score: 91", "score=91", "score 91", "budget $50,000".
            match = re.fullmatch(r"([A-Za-z][A-Za-z _/-]{1,40})\s*[:=]?\s*\$?(-?\d[\d,]*(?:\.\d+)?)", part)
            if match:
                row[_clean_key(match.group(1))] = _coerce_value(match.group(2))
                matched = True

            # Handles "4 projects", "12 tickets", "50000 budget".
            if not matched:
                match = re.fullmatch(r"\$?(-?\d[\d,]*(?:\.\d+)?)\s+([A-Za-z][A-Za-z _/-]{1,40})", part)
                if match:
                    row[_clean_key(match.group(2))] = _coerce_value(match.group(1))
                    matched = True

            if not matched:
                text_fields.append(part)

        if text_fields:
            row["descriptor"] = text_fields[0]
        if len(text_fields) > 1:
            row["outcome"] = text_fields[-1]
        for index, value in enumerate(text_fields[1:-1], start=1):
            row[f"text_feature_{index}"] = value

        if len(row) >= 2:
            rows.append(row)
    return rows or None


def parse_structured_context(context: str) -> dict[str, Any] | None:
    text = context.strip()
    if not text:
        return None
    for parser in (_parse_json_context, _parse_csv_context, _parse_plaintext_context):
        rows = parser(text)
        if rows and _enough_structure(rows):
            return {
                "data": rows,
                "focus_columns": _focus_columns(rows),
                "source_format": parser.__name__.replace("_parse_", "").replace("_context", ""),
            }
    return None
