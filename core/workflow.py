from __future__ import annotations

import asyncio
from typing import Any, TypedDict

from .logic_engine import build_analysis_tree, iter_nodes, synthesize_prediction
from .memory import MemoryManager
from .ollama_client import AsyncOllamaClient
from .persistence import AuditStore


class AnalysisWorkflowState(TypedDict, total=False):
    analysis_id: str
    request_payload: dict[str, Any]
    root_node: dict[str, Any]
    memory_matches: list[dict[str, Any]]
    memory_used: bool
    final_prediction: str
    final_confidence: float
    status: str
    workflow: str


def _build_prompt(user_goal: str, node: dict, memory_matches: list[dict]) -> str:
    memory_text = "\n".join(
        f"- {match.get('source')}: {match.get('final_prediction', '')} (score {match.get('score')})"
        for match in memory_matches[:3]
    )
    return (
        "Explain this predictive branch in plain language.\n"
        f"Goal: {user_goal}\n"
        f"Branch: {node['branch_label']} at depth {node['depth']}\n"
        f"Rows: {node['row_count']}\n"
        f"Trigger: {node.get('trigger_reason') or 'No strong trigger'}\n"
        f"Statistics: {node.get('statistical_summary', {})}\n"
        f"Similar prior cases:\n{memory_text or 'None'}\n"
        "Return a concise explanation and a practical prediction."
    )


def _fallback_response(node: dict) -> str:
    reason = node.get("trigger_reason") or "no strong trigger"
    return f"Deterministic analysis only: this branch shows {reason} across {node['row_count']} rows."


async def _explain_nodes(
    *,
    root_node: dict,
    user_goal: str,
    memory_matches: list[dict],
    use_llm: bool,
    client: AsyncOllamaClient,
) -> None:
    explainable = [node for node in iter_nodes(root_node) if node.get("trigger_reason")]
    for node in explainable:
        node["llm_prompt"] = _build_prompt(user_goal, node, memory_matches)

    if not use_llm:
        for node in explainable:
            node["llm_response"] = _fallback_response(node)
        return

    results = await asyncio.gather(
        *(client.generate(node["llm_prompt"]) for node in explainable),
        return_exceptions=True,
    )
    for node, result in zip(explainable, results):
        if isinstance(result, Exception) or not result.ok:
            node["llm_response"] = _fallback_response(node)
            node["confidence_score"] = max(0.05, round(float(node.get("confidence_score", 0)) - 0.1, 2))
        else:
            node["llm_response"] = result.text or _fallback_response(node)


def _make_nodes(*, store: AuditStore, ollama_client: AsyncOllamaClient, qdrant_url: str):
    def analyze_data(state: AnalysisWorkflowState) -> dict:
        request = state["request_payload"]
        root_node = build_analysis_tree(
            request["data"],
            user_goal=request["user_goal"],
            focus_columns=request.get("focus_columns") or [],
            max_depth=request.get("max_depth", 2),
            min_rows=request.get("min_rows", 2),
        )
        return {"root_node": root_node}

    async def search_memory(state: AnalysisWorkflowState) -> dict:
        request = state["request_payload"]
        if not request.get("use_memory", True):
            return {"memory_matches": [], "memory_used": False}
        memory = MemoryManager(store=store, ollama_client=ollama_client, qdrant_url=qdrant_url)
        matches = await memory.search(request["user_goal"])
        return {"memory_matches": matches, "memory_used": bool(matches)}

    async def explain_branches(state: AnalysisWorkflowState) -> dict:
        request = state["request_payload"]
        root_node = state["root_node"]
        await _explain_nodes(
            root_node=root_node,
            user_goal=request["user_goal"],
            memory_matches=state.get("memory_matches", []),
            use_llm=request.get("use_llm", True),
            client=ollama_client,
        )
        return {"root_node": root_node}

    def predict(state: AnalysisWorkflowState) -> dict:
        prediction, confidence = synthesize_prediction(
            state["root_node"],
            state["request_payload"]["user_goal"],
        )
        return {
            "final_prediction": prediction,
            "final_confidence": confidence,
            "status": "completed",
        }

    def persist(state: AnalysisWorkflowState) -> dict:
        store.save_analysis(
            analysis_id=state["analysis_id"],
            request_payload=state["request_payload"],
            final_prediction=state["final_prediction"],
            final_confidence=state["final_confidence"],
            root_node=state["root_node"],
            memory_used=state.get("memory_used", False),
            status=state["status"],
            memory_matches=state.get("memory_matches", []),
        )
        return {}

    async def save_semantic_memory(state: AnalysisWorkflowState) -> dict:
        request = state["request_payload"]
        if request.get("use_memory", True):
            memory = MemoryManager(store=store, ollama_client=ollama_client, qdrant_url=qdrant_url)
            await memory.save_summary(
                analysis_id=state["analysis_id"],
                text=f"{request['user_goal']}\n{state['final_prediction']}",
                final_prediction=state["final_prediction"],
            )
        return {}

    return analyze_data, search_memory, explain_branches, predict, persist, save_semantic_memory


async def _run_sequential(
    state: AnalysisWorkflowState,
    *,
    store: AuditStore,
    ollama_client: AsyncOllamaClient,
    qdrant_url: str,
) -> AnalysisWorkflowState:
    analyze_data, search_memory, explain_branches, predict, persist, save_semantic_memory = _make_nodes(
        store=store,
        ollama_client=ollama_client,
        qdrant_url=qdrant_url,
    )
    for patch in [
        analyze_data(state),
        await search_memory(state),
    ]:
        state.update(patch)
    state.update(await explain_branches(state))
    state.update(predict(state))
    persist(state)
    await save_semantic_memory(state)
    state["workflow"] = "sequential"
    return state


async def _run_langgraph(
    state: AnalysisWorkflowState,
    *,
    store: AuditStore,
    ollama_client: AsyncOllamaClient,
    qdrant_url: str,
) -> AnalysisWorkflowState:
    from langgraph.graph import END, START, StateGraph

    analyze_data, search_memory, explain_branches, predict, persist, save_semantic_memory = _make_nodes(
        store=store,
        ollama_client=ollama_client,
        qdrant_url=qdrant_url,
    )
    graph = StateGraph(AnalysisWorkflowState)
    graph.add_node("analyze_data", analyze_data)
    graph.add_node("search_memory", search_memory)
    graph.add_node("explain_branches", explain_branches)
    graph.add_node("predict", predict)
    graph.add_node("persist", persist)
    graph.add_node("save_semantic_memory", save_semantic_memory)
    graph.add_edge(START, "analyze_data")
    graph.add_edge("analyze_data", "search_memory")
    graph.add_edge("search_memory", "explain_branches")
    graph.add_edge("explain_branches", "predict")
    graph.add_edge("predict", "persist")
    graph.add_edge("persist", "save_semantic_memory")
    graph.add_edge("save_semantic_memory", END)
    result = await graph.compile().ainvoke(state)
    result["workflow"] = "langgraph"
    return result


async def run_analysis_workflow(
    *,
    analysis_id: str,
    request_payload: dict[str, Any],
    store: AuditStore,
    ollama_client: AsyncOllamaClient,
    qdrant_url: str,
) -> AnalysisWorkflowState:
    state: AnalysisWorkflowState = {
        "analysis_id": analysis_id,
        "request_payload": request_payload,
        "memory_matches": [],
        "memory_used": False,
        "status": "running",
    }
    try:
        return await _run_langgraph(
            state,
            store=store,
            ollama_client=ollama_client,
            qdrant_url=qdrant_url,
        )
    except ImportError:
        return await _run_sequential(
            state,
            store=store,
            ollama_client=ollama_client,
            qdrant_url=qdrant_url,
        )
