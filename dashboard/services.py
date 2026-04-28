from __future__ import annotations

from typing import Any, Dict

from django.conf import settings

from explain.pipeline import ExplainerPipeline
from llm import create_client


def current_runtime_config(form_data: Dict[str, Any] | None = None) -> Dict[str, Any]:
    base = dict(settings.BBE_SETTINGS)
    if form_data:
        if form_data.get("base_url"):
            base["base_url"] = form_data["base_url"].strip()
        if form_data.get("model"):
            base["model"] = form_data["model"].strip()
    return base


def backend_status(config: Dict[str, Any]) -> tuple[bool, str]:
    if not config["base_url"]:
        return False, "Add your Ollama URL."
    if not config["model"]:
        return False, "Add your Ollama model name."

    client = create_client(
        base_url=config["base_url"],
        model=config["model"],
        timeout_seconds=config["timeout_seconds"],
    )
    return client.healthcheck()


def run_explainer(*, question: str, context: str, config: Dict[str, Any]) -> Dict[str, Any]:
    client = create_client(
        base_url=config["base_url"],
        model=config["model"],
        timeout_seconds=config["timeout_seconds"],
    )
    pipeline = ExplainerPipeline(client)
    result = pipeline.run(
        question=question,
        context=context,
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        critique_pass=config["critique_pass"],
    )
    result["question"] = question
    result["context"] = context
    result["base_url"] = config["base_url"]
    result["model"] = config["model"]
    return result
