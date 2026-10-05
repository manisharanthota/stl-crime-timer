"""Match all unlinked classifications into incidents once: `python -m matcher`."""

import logging

from matcher.match import match_pending


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from db import SessionLocal

    with SessionLocal() as session:
        counts = match_pending(session)
    for name, count in counts.items():
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
