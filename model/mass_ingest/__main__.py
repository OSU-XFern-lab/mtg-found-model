"""Run one stage of the bulk-ingestion workflow."""

from __future__ import annotations

import argparse
import os


STAGES = ("plan", "collect", "dedup", "embed", "vq", "build", "publish", "cleanup")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m mass_ingest", description=__doc__)
    parser.add_argument("stage", choices=STAGES + ("audit",))
    parser.add_argument(
        "audit_stage", nargs="?", metavar="COMPLETED_STAGE",
        choices=("collect", "dedup", "embed", "vq", "build"),
        help="stage to verify when the command is audit",
    )
    parser.add_argument("--task", type=int, help="task or bucket index for an array stage")
    args = parser.parse_args()
    if args.task is not None:
        os.environ["TASK_ID"] = str(args.task)
    if args.stage == "audit":
        if not args.audit_stage:
            parser.error("audit requires a completed stage name")
        from .audit import main as run
        run(args.audit_stage)
        return
    if args.audit_stage:
        parser.error("a second stage name is only valid with audit")
    if args.stage in ("plan", "collect", "dedup"):
        from .prepare import main as run
    elif args.stage in ("embed", "vq"):
        from .gpu import main as run
    elif args.stage in ("build", "publish"):
        from .build import main as run
    else:
        from .cleanup import main as run
    run(args.stage) if args.stage != "cleanup" else run()


if __name__ == "__main__":
    main()
