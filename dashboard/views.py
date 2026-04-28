from __future__ import annotations

from django.conf import settings
from django.contrib import messages
from django.shortcuts import render

from .forms import ExplainForm
from .services import backend_status, current_runtime_config, run_explainer
from .storage import MongoRunStore


def _store() -> MongoRunStore:
    return MongoRunStore(
        uri=settings.BBE_SETTINGS["mongodb_uri"],
        database_name=settings.BBE_SETTINGS["mongodb_db"],
    )


def home(request):
    defaults = current_runtime_config()
    result = None

    if request.method == "POST":
        form = ExplainForm(request.POST)
        if form.is_valid():
            runtime = current_runtime_config(form.cleaned_data)
            ready, status = backend_status(runtime)
            if not ready:
                messages.error(request, status)
            else:
                try:
                    result = run_explainer(
                        question=form.cleaned_data["question"].strip(),
                        context=form.cleaned_data["context"],
                        config=runtime,
                    )
                    saved = _store().save_run(result)
                    if saved:
                        messages.success(request, "Run saved to MongoDB history.")
                    else:
                        messages.info(request, "Run completed. MongoDB history is not active.")
                except Exception as exc:
                    messages.error(request, f"Explainer failed: {exc}")
        else:
            messages.error(request, "Fix the form fields and try again.")
    else:
        form = ExplainForm(
            initial={
                "base_url": defaults["base_url"],
                "model": defaults["model"],
            }
        )

    runtime = current_runtime_config(form.cleaned_data if form.is_bound and form.is_valid() else None)
    ready, status = backend_status(runtime)
    recent_runs = _store().recent_runs(limit=settings.BBE_SETTINGS["history_limit"])

    context = {
        "form": form,
        "result": result,
        "recent_runs": recent_runs,
        "runtime": runtime,
        "backend_ready": ready,
        "backend_status": status,
        "mongo_enabled": bool(settings.BBE_SETTINGS["mongodb_uri"]),
    }
    return render(request, "dashboard/home.html", context)
