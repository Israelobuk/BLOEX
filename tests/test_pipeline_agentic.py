from explain.pipeline import ExplainerPipeline


class WeakClient:
    def __init__(self):
        self.timeout_seconds = 5
        self.calls = 0

    def chat(self, messages, temperature, max_tokens, timeout_seconds=None):
        self.calls += 1
        return "ANSWER: weak answer\nBLACK_BOX: weak explanation"

    def metadata(self):
        return {"backend": "test"}


def test_run_agentic_stops_after_loop_budget_on_weak_output():
    pipeline = ExplainerPipeline(WeakClient())

    result = pipeline.run_agentic(
        question="Is this answer supported?",
        model_answer="Yes, definitely.",
        context="Source text is thin and incomplete.",
        temperature=0.0,
        max_tokens=64,
        completion_retry=False,
        max_agent_loops=2,
    )

    assert result["answer"]
    assert result["black_box_explanation"]
    assert result["confidence_reason"]
