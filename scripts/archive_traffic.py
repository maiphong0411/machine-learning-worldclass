"""Merge GitHub's last-14-days traffic into CSV files that keep the full history.

Usage: GH_TOKEN=... REPO=owner/name python3 scripts/archive_traffic.py <data_dir>

Writes (one row per day, newest values win for overlapping days):
  views.csv   date,views,unique_visitors
  clones.csv  date,clones,unique_cloners
and keeps one snapshot per day of of the top pages and referrers:
  paths.csv      snapshot_date,path,title,views,unique_visitors
  referrers.csv  snapshot_date,referrer,views,unique_visitors
"""

import csv
import datetime as dt
import json
import os
import sys
import urllib.request
from pathlib import Path

API = "https://api.github.com/repos/{repo}/traffic/{endpoint}"


def get(endpoint: str):
    req = urllib.request.Request(
        API.format(repo=os.environ["REPO"], endpoint=endpoint),
        headers={
            "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def merge_daily(path: Path, header: list[str], rows: list[dict]) -> None:
    existing = {}
    if path.exists():
        with path.open() as f:
            existing = {r["date"]: r for r in csv.DictReader(f)}
    for r in rows:
        day = r["timestamp"][:10]
        existing[day] = {"date": day, header[1]: r["count"], header[2]: r["uniques"]}
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(existing[d] for d in sorted(existing))


def append_snapshot(path: Path, header: list[str], rows: list[list]) -> None:
    """Add today's rows, replacing any earlier snapshot from the same day (re-runs are safe)."""
    kept = []
    if path.exists():
        with path.open() as f:
            kept = [r for r in list(csv.reader(f))[1:] if r and r[0] != str(rows[0][0] if rows else "")]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(kept + rows)


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    merge_daily(out / "views.csv", ["date", "views", "unique_visitors"], get("views")["views"])
    merge_daily(out / "clones.csv", ["date", "clones", "unique_cloners"], get("clones")["clones"])
    append_snapshot(out / "paths.csv", ["snapshot_date", "path", "title", "views", "unique_visitors"],
                    [[today, p["path"], p["title"], p["count"], p["uniques"]] for p in get("popular/paths")])
    append_snapshot(out / "referrers.csv", ["snapshot_date", "referrer", "views", "unique_visitors"],
                    [[today, r["referrer"], r["count"], r["uniques"]] for r in get("popular/referrers")])
    print(f"Traffic archived to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
