"""Require completed full miniexpr qualification for an exact native revision."""

import argparse
import json
import os
import re
import urllib.request


def github_get(path):
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(f"https://api.github.com/repos/Blosc/miniexpr/{path}", headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def verify(native_sha, get=github_get):
    if not re.fullmatch(r"[0-9a-f]{40}", native_sha):
        raise ValueError("Native qualification requires an exact, full lowercase commit SHA")
    runs = get(f"actions/workflows/ci.yml/runs?head_sha={native_sha}&status=success&per_page=100")
    for run in runs["workflow_runs"]:
        if run["head_sha"] != native_sha or run["conclusion"] != "success":
            continue
        page = 1
        while True:
            jobs = get(f"actions/runs/{run['id']}/jobs?filter=latest&per_page=100&page={page}")
            if any(
                job["name"] == "Publish full native qualification"
                and job["status"] == "completed"
                and job["conclusion"] == "success"
                for job in jobs["jobs"]
            ):
                return {
                    "native_sha": native_sha,
                    "run_id": run["id"],
                    "url": run["html_url"],
                    "tier": "full",
                }
            if page * 100 >= jobs["total_count"]:
                break
            page += 1
    raise RuntimeError(
        f"No successful full miniexpr qualification for {native_sha}. "
        "Dispatch native CI at that revision with full_qualification=true first."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_sha")
    parser.add_argument("--output", default="native-qualification.json")
    args = parser.parse_args()
    evidence = verify(args.native_sha)
    with open(args.output, "w") as output:
        json.dump(evidence, output, indent=2)
        output.write("\n")
    print(f"Qualified native revision {args.native_sha}: {evidence['url']}")


if __name__ == "__main__":
    main()
