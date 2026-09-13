"""Archive structured probe evidence and print compact ablation aggregates."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = []
    groups = defaultdict(list)
    for path in args.inputs:
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        evidence.append(
            dict(name=path.name, sha256=hashlib.sha256(raw).hexdigest(), rows=rows)
        )
        for row in rows:
            if row.get("event") == "summary":
                groups[(row["kind"], row["size"], row["mode"])].append(row)
    summaries = []
    for (kind, size, mode), rows in groups.items():
        summary = dict(
            kind=kind,
            size=size,
            mode=mode,
            repeats=len(rows),
            completed=[row["completed"] for row in rows],
            seconds_mean=mean(row["seconds"] for row in rows),
            throughput_mean=mean(row["throughput"] for row in rows),
            peak_gib=max(row["peak_bytes"] for row in rows) / 1024**3,
            final_batches=[row["final_batch"] for row in rows],
            oom_count=sum(row["oom_count"] for row in rows),
        )
        summaries.append(summary)
        print(json.dumps(summary))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(dict(summaries=summaries, evidence=evidence), output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
