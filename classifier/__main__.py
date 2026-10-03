"""Classify all new raw items once: `python -m classifier`."""

import logging

from classifier.classify import classify_pending


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from db import SessionLocal

    with SessionLocal() as session:
        counts = classify_pending(session)
    stopped = counts.pop("stopped_quota", 0)
    for name, count in counts.items():
        print(f"{name}: {count}")
    if stopped:
        print("Stopped early: LLM daily quota exhausted; remaining items left new.")


if __name__ == "__main__":
    main()
