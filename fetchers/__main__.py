"""Run all fetchers once: `python -m fetchers`."""

import logging

from fetchers.runner import run_fetchers


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from db import SessionLocal

    with SessionLocal() as session:
        results = run_fetchers(session)
    for name, added in results.items():
        print(f"{name}: {added} new item(s)")
    print(f"{len(results)} source(s) fetched successfully")


if __name__ == "__main__":
    main()
