"""Bounded diagnostics on usage records, not inferred provider cache internals."""
from collections import Counter


def add_findings(report: dict) -> dict:
    """Annotate an accounting report without attributing unobserved savings."""
    findings = []
    previous = {}
    fingerprinted = 0
    for request in report["requests"]:
        fingerprint = request.get("prefix_fingerprint")
        key = (request["task_id"], request["session_id"])
        if fingerprint:
            fingerprinted += 1
            before = previous.get(key)
            if before and before.get("prefix_fingerprint") and before["prefix_fingerprint"] != fingerprint:
                reason = request.get("prefix_change_reason", "unknown")
                findings.append({
                    "code": "declared_prefix_changed",
                    "task_id": request["task_id"],
                    "request_id": request["request_id"],
                    "previous_request_id": before["request_id"],
                    "declared_reason": reason,
                    "observation": "The exporter supplied different fingerprints for the designated stable prefix.",
                    "interpretation": "This is not proof of a cache miss or of avoidable billed work.",
                    "next_step": "Check whether the edit was necessary; use provider cache diagnostics before changing the request builder.",
                })
        # A gap in fingerprint coverage must not masquerade as adjacent evidence.
        previous[key] = request

    totals = report["totals"]
    if any(request["kind"] == "compaction" for request in report["requests"]):
        findings.append({
            "code": "compaction_cost_included",
            "observation": "Compaction requests are included in total cost, rather than treated as free.",
            "cost_usd": totals["compaction_request_cost_usd"],
            "next_step": "Compare the whole completed task; shorter later prompts alone do not establish net savings.",
        })
    failed = [task["task_id"] for task in report["tasks"] if not task["success"]]
    if failed:
        findings.append({
            "code": "failed_work_included",
            "task_ids": failed,
            "observation": "Requests belonging to unsuccessful tasks remain in total cost.",
            "next_step": "Check task outcomes before accepting any lower cost per successful task.",
        })
    report["findings"] = findings
    report["coverage"] = {
        "requests_with_prefix_fingerprint": fingerprinted,
        "requests": len(report["requests"]),
        "tasks_with_duration": sum(task.get("duration_ms") is not None for task in report["tasks"]),
        "tasks": len(report["tasks"]),
        "finding_counts": dict(Counter(finding["code"] for finding in findings)),
    }
    report["diagnostic_limits"] = [
        "Fingerprints describe a caller-designated stable prefix; they do not prove a provider cache key or residency.",
        "No prefix fingerprint means no structural prefix diagnosis, not an unchanged prefix.",
        "Source counts and terminal events detect malformed exports, not records omitted by the exporter before export.",
        "Reported usage multiplied by supplied rates is an accounting calculation, not an independently reconciled invoice.",
        "Task success labels come from the supplied evaluator; this tool does not judge answers or rerun the agent.",
        "No automatic prompt rewriting, provider requests, cache mutation or guaranteed savings are performed.",
    ]
    return report
