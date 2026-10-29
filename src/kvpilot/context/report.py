"""Portable reports with escaped user-controlled labels and no external assets."""
import html
import json
from decimal import Decimal
from pathlib import Path


def _money(value):
    return "unavailable" if value is None else f"${Decimal(str(value)):.6f}"


def _cell(value):
    if value is None:
        return "unavailable"
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _sections(report):
    """Return plain text sections and tables; HTML and Markdown share the facts."""
    kind = report.get("evidence_type", "unknown")
    if kind == "scenario_estimate":
        return "Compaction break-even estimate", [
            ("What this means", ["This is a hypothetical calculation, not a measurement of savings or answer quality.", report["equation"]]),
            ("Result", [("Keep modeled context", _money(report["keep_cost_usd"])),
                        ("Compact modeled context", _money(report["compact_cost_usd"])),
                        ("Estimated net saving", _money(report["estimated_net_savings_usd"])),
                        ("First strictly cheaper future turn", report["first_strictly_cheaper_turn"])]),
            ("Assumptions", report["assumptions"]),
        ], []
    if "totals" in report:
        totals = report["totals"]
        sections = [
            ("What this means", [
                "Synthetic example with illustrative prices. This is not a customer saving." if kind == "synthetic_example"
                else "Accounting from supplied usage and prices. This is not an independent invoice audit or a quality evaluation.",
                "All recorded attempts and compaction requests count toward cost, including unsuccessful tasks.",
            ]),
            ("Recorded costs", [("Total model usage cost", _money(totals["total_cost_usd"])),
                                ("Successful tasks", totals["successful_tasks"]),
                                ("Unsuccessful tasks", totals["failed_tasks"]),
                                ("Cost per successful task", _money(totals["cost_per_successful_task_usd"])),
                                ("Compaction request cost (already included)", _money(totals["compaction_request_cost_usd"]))]),
            ("What to investigate", [finding["observation"] + " " + finding["next_step"]
                                     for finding in report.get("findings", [])]
             or ["No structural finding from the available fields. This does not prove the workflow is optimal."]),
            ("Limits", report.get("diagnostic_limits", []) + report.get("scope_exclusions", [])),
            ("Provenance", [("Run", report["metadata"]["run_id"]),
                            ("Trace SHA256", report["trace_sha256"]),
                            ("Price source", report["pricing"]["source"]),
                            ("Price effective date", report["pricing"]["effective_date"])]),
        ]
        tables = [("Task outcomes", ["Task", "Supplied outcome", "Usage cost", "Task wall time (ms)"],
                   [[task["task_id"], "success" if task["success"] else "failure", _money(task["total_cost_usd"]), task.get("duration_ms")]
                    for task in report["tasks"]])]
        return "Agent context cost audit", sections, tables
    status = report.get("status", "inconclusive")
    sections = [("Decision", [f"Comparison status: {status}.",
                             "This checks supplied outcomes and descriptive costs, not statistical significance or general answer quality.",
                             "Synthetic fixture: a pass is only a software example, never customer evidence."
                             if kind == "synthetic_example" else "A pass does not independently establish customer savings."]),
                ("Runs", [("Baseline", report["baseline_run_id"]), ("Candidate", report["candidate_run_id"])]),
                ("Why", report.get("reasons") or ["All declared comparison gates passed on the supplied records."]),
                ("Limits", report.get("limitations", []))]
    tables = [("Regression gates", ["Check", "Result", "Explanation"],
               [[name, gate["status"], "; ".join(gate["reasons"]) or "Threshold satisfied on supplied records"]
                for name, gate in report["gates"].items()])]
    summary = report.get("summary")
    if summary:
        pct = lambda value: "unavailable" if value is None else f"{Decimal(str(value)):.2f}%"
        tables.insert(0, ("Paired run totals", ["Metric", "Baseline", "Candidate"], [
            ["All recorded model usage", _money(summary["baseline_total_cost_usd"]), _money(summary["candidate_total_cost_usd"])],
            ["Successful tasks", summary["baseline_successful_tasks"], summary["candidate_successful_tasks"]],
            ["Unsuccessful tasks", summary["baseline_failed_tasks"], summary["candidate_failed_tasks"]],
            ["Cost per successful task", _money(summary["baseline_cost_per_successful_task_usd"]), _money(summary["candidate_cost_per_successful_task_usd"])],
            ["p95 task wall time (ms)", summary["baseline_p95_task_duration_ms"], summary["candidate_p95_task_duration_ms"]],
        ]))
        sections.insert(2, ("Descriptive change", [
            ("Cost-per-success savings (positive is cheaper)", pct(summary["cost_per_success_savings_pct"])),
            ("p95 task duration change (negative is faster)", pct(summary["p95_task_duration_regression_pct"])),
            ("Previously successful tasks now failing", ", ".join(report["paired_outcomes"]["regressions"]) or "none"),
        ]))
    return "Agent context cost comparison", sections, tables


def render_markdown(report):
    title, sections, tables = _sections(report)
    parts = [f"# {title}\n", f"Evidence: `{_cell(report.get('evidence_type', 'unknown'))}`\n"]
    for heading, items in sections:
        parts.append(f"## {heading}\n")
        for item in items:
            if isinstance(item, tuple):
                parts.append(f"- **{html.escape(_cell(item[0]))}:** {html.escape(_cell(item[1]))}")
            elif heading == "Results":
                # Escape Markdown fence delimiters in source labels too.
                parts.append("<pre>" + html.escape(item) + "</pre>")
            else:
                parts.append("- " + html.escape(_cell(item)))
        parts.append("")
    for heading, columns, rows in tables:
        parts.extend([f"## {heading}\n", "| " + " | ".join(columns) + " |",
                      "| " + " | ".join("---" for _ in columns) + " |"])
        parts.extend("| " + " | ".join(html.escape(_cell(cell)) for cell in row) + " |" for row in rows)
        parts.append("")
    return "\n".join(parts) + "\n"


def render_html(report):
    title, sections, tables = _sections(report)
    e = lambda value: html.escape("unavailable" if value is None else str(value), quote=True)
    body = [f"<header><p class='eyebrow'>KVPILOT · LOCAL REPORT</p><h1>{e(title)}</h1>",
            f"<p class='badge'>{e(report.get('evidence_type', 'unknown'))}</p></header>"]
    for heading, items in sections:
        body.append(f"<section><h2>{e(heading)}</h2>")
        for item in items:
            if isinstance(item, tuple):
                body.append(f"<p><strong>{e(item[0])}</strong><span class='value'>{e(item[1])}</span></p>")
            elif heading == "Results":
                body.append(f"<pre>{e(item)}</pre>")
            else:
                body.append(f"<p>{e(item)}</p>")
        body.append("</section>")
    for heading, columns, rows in tables:
        body.append(f"<section><h2>{e(heading)}</h2><div class='scroll'><table><thead><tr>")
        body.extend(f"<th>{e(column)}</th>" for column in columns)
        body.append("</tr></thead><tbody>")
        for row in rows:
            body.append("<tr>" + "".join(f"<td>{e(cell)}</td>" for cell in row) + "</tr>")
        body.append("</tbody></table></div></section>")
    return """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>""" + e(title) + """</title><style>
:root{font-family:system-ui,sans-serif;color:#172b36;background:#f3f6f5;line-height:1.65}
body{max-width:1000px;margin:auto;padding:36px 22px}h1{font-size:clamp(28px,5vw,42px);line-height:1.15}
h2{font-size:20px;color:#17685a}.eyebrow{font-size:12px;letter-spacing:.14em;color:#17685a}
.badge{display:inline-block;background:#e2ebe7;padding:5px 12px;border-radius:6px;font-family:monospace}
section{background:white;border:1px solid #d9e3dd;border-radius:12px;padding:20px 26px;margin:18px 0}
.value{display:block;font-family:monospace;font-size:18px;overflow-wrap:anywhere}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.scroll{overflow:auto}
table{border-collapse:collapse;width:100%;text-align:left}th,td{padding:10px;border-bottom:1px solid #e4ebe7;overflow-wrap:anywhere}
</style><body>""" + "".join(body) + "</body></html>\n"


def write_bundle(report, directory):
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    (target / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (target / "report.md").write_text(render_markdown(report), encoding="utf-8")
    (target / "report.html").write_text(render_html(report), encoding="utf-8")
    return target
