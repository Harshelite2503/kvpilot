"""Local context-usage accounting; importing this module makes no API calls."""
import json
import sys

from .analysis import add_findings
from .billing import audit_costs
from .compare import compare_reports
from .plan import estimate_compaction
from .report import write_bundle
from .schema import load_pricing, load_trace


def add_parser(subparsers):
    parser = subparsers.add_parser("context", help="local agent context-cost audit and comparison")
    commands = parser.add_subparsers(dest="context_command", required=True)
    audit = commands.add_parser("audit", help="account for exported usage with an explicit price card")
    audit.add_argument("--trace", required=True)
    audit.add_argument("--pricing", required=True)
    audit.add_argument("--output", required=True, help="directory for report.json, report.md and report.html")
    compare = commands.add_parser("compare", help="compare identical tasks and evaluate regression gates")
    compare.add_argument("--baseline", required=True, help="baseline JSONL trace")
    compare.add_argument("--candidate", required=True, help="candidate JSONL trace")
    compare.add_argument("--pricing", required=True)
    compare.add_argument("--min-savings-pct", type=float, default=0)
    compare.add_argument("--max-p95-regression-pct", type=float, default=5)
    compare.add_argument("--fail-on-regression", action="store_true", help="exit 1 if gates fail or evidence is inconclusive")
    compare.add_argument("--output", required=True)
    plan = commands.add_parser("plan", help="hypothetical compaction break-even, separate from measured usage")
    plan.add_argument("--current-tokens", type=int, required=True)
    plan.add_argument("--retained-tokens", type=int, required=True)
    plan.add_argument("--future-turns", type=int, required=True)
    plan.add_argument("--cache-read-usd-per-million", required=True)
    plan.add_argument("--rewrite-cost-usd", required=True)
    plan.add_argument("--summary-cost-usd", required=True)
    plan.add_argument("--output", required=True)


def run(args):
    try:
        if args.context_command == "audit":
            report = add_findings(audit_costs(load_trace(args.trace), load_pricing(args.pricing)))
        elif args.context_command == "compare":
            pricing = load_pricing(args.pricing)
            baseline = audit_costs(load_trace(args.baseline), pricing)
            candidate = audit_costs(load_trace(args.candidate), pricing)
            report = compare_reports(baseline, candidate, min_savings_pct=args.min_savings_pct,
                                     max_p95_regression_pct=args.max_p95_regression_pct)
        else:
            report = estimate_compaction(
                current_tokens=args.current_tokens, retained_tokens=args.retained_tokens,
                future_turns=args.future_turns, cache_read_usd_per_million=args.cache_read_usd_per_million,
                rewrite_cost_usd=args.rewrite_cost_usd, summary_cost_usd=args.summary_cost_usd)
        target = write_bundle(report, args.output)
    except (ValueError, OSError) as exc:
        print(f"context: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    brief = {"output": str(target), "evidence_type": report["evidence_type"]}
    if "status" in report:
        brief.update(status=report["status"], passed=report["passed"], certifies_customer_savings=False)
    elif "totals" in report:
        brief["totals"] = report["totals"]
    else:
        brief["estimated_net_savings_usd"] = report["estimated_net_savings_usd"]
        brief["first_strictly_cheaper_turn"] = report["first_strictly_cheaper_turn"]
    print(json.dumps(brief, indent=2, allow_nan=False))
    if args.context_command == "compare" and args.fail_on_regression and not report["passed"]:
        raise SystemExit(1)
