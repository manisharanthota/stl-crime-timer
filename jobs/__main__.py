"""Job runner CLI: `python -m jobs once` or `python -m jobs schedule`."""

import argparse

from jobs.logsetup import setup_logging
from jobs.scheduler import run_once, run_scheduled


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m jobs")
    parser.add_argument(
        "command",
        choices=["once", "schedule"],
        help="once: run the pipeline one time; schedule: run it every PIPELINE_INTERVAL_MINUTES",
    )
    args = parser.parse_args(argv)
    setup_logging()
    if args.command == "once":
        run_once()
    else:
        run_scheduled()


if __name__ == "__main__":
    main()
