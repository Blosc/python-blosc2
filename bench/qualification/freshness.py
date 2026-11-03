"""Read-only scheduled-result watchdog. No issue publication or secret storage."""

import datetime as dt
import json
import os
import urllib.request
from pathlib import Path


def complete_run(run, jobs):
    return (
        run["conclusion"] == "success"
        and run["display_title"] == "Menudet perf observation"
        and len(jobs) == 2
        and all(j["conclusion"] == "success" for j in jobs)
    )


def stale(timestamp, now, hours=48):
    return (
        timestamp is None
        or (now - dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))).total_seconds() > hours * 3600
    )


def main():
    repository, branch = os.environ["GITHUB_REPOSITORY"], os.environ["DEFAULT_BRANCH"]

    def get(path):
        request = urllib.request.Request(
            f"https://api.github.com/repos/{repository}/{path}",
            headers={
                "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
                "Accept": "application/vnd.github+json",
            },
        )
        # No retry loop: on rate limits, fail this check without issuing more requests.
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    from urllib.parse import quote

    runs = get(
        f"actions/workflows/perf-qualification-menudet.yml/runs?branch={quote(branch, safe='')}&per_page=100"
    )["workflow_runs"]
    latest = None
    for run in runs:
        if run["conclusion"] != "success" or run["display_title"] != "Menudet perf observation":
            continue
        jobs = get(f"actions/runs/{run['id']}/jobs?per_page=100")["jobs"]
        if complete_run(run, jobs):
            latest = run
            break
    timestamp = latest["updated_at"] if latest else None
    result = {
        "last_complete_run": latest["id"] if latest else None,
        "revision": latest["head_sha"] if latest else None,
        "timestamp": timestamp,
        "stale": stale(timestamp, dt.datetime.now(dt.UTC)),
        "limitation": "GitHub-hosted watchdog shares the scheduler failure domain; not independent monitoring.",
    }
    Path("menudet-freshness.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if result["stale"]:
        raise SystemExit("No complete routine performance qualification in the past 48 hours")


if __name__ == "__main__":
    main()
