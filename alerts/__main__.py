"""Alerts CLI: `python -m alerts test` sends a test message (ignores cooldowns)."""

import argparse
import sys
from datetime import datetime, timezone

from alerts import notify
from config import get_settings
from jobs.logsetup import setup_logging
from timeutil import to_local


def main(argv: list[str] | None = None, send=notify.send) -> int:
    parser = argparse.ArgumentParser(prog="python -m alerts")
    parser.add_argument("command", choices=["test"], help="test: send a test alert")
    parser.parse_args(argv)
    setup_logging()

    now = to_local(datetime.now(timezone.utc)).strftime("%b %d %I:%M %p")
    if not send(f"✅ STL crime tracker test alert ({now}). Alerts are working."):
        print("Test alert failed; see the log above.")
        return 1
    if get_settings().alert_webhook_url:
        print("Test alert sent.")
    else:
        print("ALERT_WEBHOOK_URL is not set: the alert was only logged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
