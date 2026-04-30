from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    user_goal: str = Field(min_length=1)
    data: list[dict[str, Any]] = Field(min_length=1)
    focus_columns: list[str] = Field(default_factory=list)
    max_depth: int = Field(default=2, ge=0, le=5)
    min_rows: int = Field(default=2, ge=1)
    use_memory: bool = True
    use_llm: bool = True


class AnalysisNode(BaseModel):
    node_id: str
    parent_id: str | None
    depth: int
    branch_label: str
    row_count: int
    statistical_summary: dict[str, Any]
    trigger_reason: str
    llm_prompt: str
    llm_response: str
    confidence_score: float
    children: list["AnalysisNode"] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    analysis_id: str
    final_prediction: str
    final_confidence: float
    root_node: AnalysisNode
    memory_used: bool
    status: str


class HistoryResponse(BaseModel):
    items: list[dict[str, Any]]


AnalysisNode.model_rebuild()
