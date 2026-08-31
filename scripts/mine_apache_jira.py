"""Mine Apache's public JIRA for issues that record BOTH an estimate and an actual.

WHY. `postg0_sip_within_org.md` asked whether an organisation's own estimation error is
learnable from its own history, and answered no — on ONE organisation. The obvious
objection is that SiP is a single company with a single logging culture. Apache's JIRA
is the cheapest way to ask the same question of many independent teams: it is public, it
needs no credentials, and every issue that carries `timeoriginalestimate` *and*
`timespent` is a recorded human estimate paired with a recorded actual, dated, with the
project as a natural grouping.

WHAT IT IS NOT. This is a small corpus — the whole of Apache JIRA yields about two
thousand such issues across ~180 projects, because time tracking is optional and most
contributors ignore it. It is fresh and independent, not large. Treat the per-project
numbers as underpowered and read the pooled paired test.

Output: data/raw/apache_jira/issues.csv (not committed — regenerate with this script).

    python scripts/mine_apache_jira.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://issues.apache.org/jira/rest/api/2/search"
# Only issues carrying both halves of the pair. Ordered by creation so a rerun that
# stops early still holds a prefix of the same corpus rather than a random sample.
JQL = "timespent > 0 AND timeoriginalestimate > 0 ORDER BY created ASC"
FIELDS = (
    "project,timespent,timeoriginalestimate,timeestimate,created,resolutiondate,"
    "issuetype,priority,reporter,assignee,summary,components"
)
PAGE = 200
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "apache_jira" / "issues.csv"

COLUMNS = (
    "key", "project", "created", "resolved", "issuetype", "priority",
    "reporter", "assignee", "n_components", "summary",
    "estimate_seconds", "spent_seconds",
)


def _get(start: int) -> dict:
    query = urllib.parse.urlencode(
        {"jql": JQL, "fields": FIELDS, "maxResults": PAGE, "startAt": start}
    )
    request = urllib.request.Request(
        f"{BASE}?{query}",
        # A public read-only endpoint, but identify the client anyway: an anonymous
        # scraper with no user agent is the first thing a rate limiter drops.
        headers={"User-Agent": "metis-benchmark/1.0 (research; info@nacodestudios.it)"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _row(issue: dict) -> dict:
    f = issue["fields"]

    def name(field: str) -> str:
        value = f.get(field) or {}
        return value.get("name") or value.get("key") or ""

    return {
        "key": issue["key"],
        "project": f["project"]["key"],
        "created": f.get("created") or "",
        "resolved": f.get("resolutiondate") or "",
        "issuetype": name("issuetype"),
        "priority": name("priority"),
        # Reporter and assignee are the JIRA equivalents of SiP's RaisedByID /
        # AssignedToID: known when the estimate is made, so admissible features.
        "reporter": name("reporter"),
        "assignee": name("assignee"),
        "n_components": len(f.get("components") or []),
        "summary": (f.get("summary") or "").replace("\n", " "),
        "estimate_seconds": f.get("timeoriginalestimate") or 0,
        "spent_seconds": f.get("timespent") or 0,
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    start = 0
    while True:
        payload = _get(start)
        issues = payload.get("issues", [])
        if not issues:
            break
        rows.extend(_row(i) for i in issues)
        start += len(issues)
        print(f"  {start}/{payload['total']}", flush=True)
        if start >= payload["total"]:
            break
        # Deliberate throttle: this is someone else's public infrastructure and the
        # whole corpus is eleven requests.
        time.sleep(1.0)

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    projects = len({r["project"] for r in rows})
    print(f"wrote {OUT}: {len(rows)} issues across {projects} projects")


if __name__ == "__main__":
    sys.exit(main())
