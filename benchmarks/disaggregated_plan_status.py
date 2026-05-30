from __future__ import annotations

import argparse
import json
from pathlib import Path

from stageml.disaggregated_execution import disaggregated_status


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper_outputs/disaggregated_plan_status.json")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    status = disaggregated_status()
    out.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    with out.with_suffix(".csv").open("w", encoding="utf-8") as f:
        f.write("plan_kind,status,claim_boundary\n")
        f.write(f"{status['plan_kind']},{status['status']},{status['claim_boundary']}\n")
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
