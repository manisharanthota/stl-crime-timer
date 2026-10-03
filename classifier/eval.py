"""Score the real classifier against hand-labeled headlines: `python -m classifier.eval`.

Calls the live LLM (needs GEMINI_API_KEY) in batches, caching results in
.eval_cache.json so reruns don't spend quota. Run manually; pytest does not collect it.
"""

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import yaml

from classifier.classify import (
    BATCH_SIZE,
    LLMUnavailable,
    QuotaExhausted,
    classify_batch,
    make_limiter,
)
from classifier.llm import get_llm_client
from classifier.prefilter import prefilter
from classifier.prompt import PROMPT_VERSION
from classifier.schema import ClassifierOutput
from models import RawItem

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "tests" / "fixtures" / "eval_headlines.yaml"
DEFAULT_CACHE = ROOT / ".eval_cache.json"
FIELDS = ("is_crime", "crime_type", "in_stl")


def load_cases(path: Path) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_cache(path: Path) -> dict[str, dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def save_cache(path: Path, cache: dict[str, dict]) -> None:
    path.write_text(json.dumps(cache, indent=1, sort_keys=True), encoding="utf-8")


def cache_key(case: dict, model: str) -> str:
    """Changes whenever the model, prompt version, or case text changes."""
    parts = [
        model, PROMPT_VERSION, case["title"], case.get("body") or "", case.get("published_at") or ""
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def to_item(case: dict, item_id: int) -> RawItem:
    published = case.get("published_at")
    return RawItem(
        id=item_id,
        title=case["title"],
        body=case.get("body"),
        published_at=datetime.fromisoformat(published) if published else None,
    )


def predict_all(
    cases: list[dict], llm, cache: dict[str, dict], use_cache: bool, sleep=time.sleep
) -> tuple[dict[int, ClassifierOutput], str | None]:
    """Returns ({case index: prediction}, error message if the run stopped early).
    Cases that fail the prefilter are predicted not-crime without the LLM; new LLM
    results are written into `cache`."""
    preds: dict[int, ClassifierOutput] = {}
    pending: list[RawItem] = []
    for i, case in enumerate(cases):
        item = to_item(case, i)
        key = cache_key(case, llm.model)
        if not prefilter(item):
            preds[i] = ClassifierOutput(is_crime=False, in_stl=False, confidence=1.0)
        elif use_cache and key in cache:
            preds[i] = ClassifierOutput.model_validate(cache[key])
        else:
            pending.append(item)

    limiter = make_limiter(sleep)
    for start in range(0, len(pending), BATCH_SIZE):
        batch = pending[start : start + BATCH_SIZE]
        try:
            results = classify_batch(batch, llm, limiter, sleep)
        except QuotaExhausted:
            return preds, "daily quota exhausted"
        except LLMUnavailable as exc:
            return preds, f"LLM unavailable: {exc}"
        for item in batch:
            if item.id in results:
                fields = results[item.id].model_dump(exclude={"raw_item_id"})
                out = ClassifierOutput.model_validate(fields)
                preds[item.id] = out
                cache[cache_key(cases[item.id], llm.model)] = out.model_dump(mode="json")
    return preds, None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--no-cache", action="store_true",
        help="ignore cached results (fresh results are still saved)",
    )
    args = parser.parse_args(argv)

    cases = load_cases(args.path)
    llm = get_llm_client()
    cache = load_cache(args.cache)
    preds, error = predict_all(cases, llm, cache, use_cache=not args.no_cache)
    save_cache(args.cache, cache)

    field_hits = {f: 0 for f in FIELDS}
    field_total = {f: 0 for f in FIELDS}
    rows_ok = scored_rows = 0
    for i, case in enumerate(cases):
        expected = case["expected"]
        title = case["title"][:70]
        pred = preds.get(i)
        if pred is None:
            print(f"SKIP {i + 1:2d}. {title:<70} no result")
            continue
        scored_rows += 1
        misses = []
        for f in FIELDS:
            want = expected.get(f)
            if f == "in_stl" and want is None:
                continue
            field_total[f] += 1
            got = getattr(pred, f)
            if got == want:
                field_hits[f] += 1
            else:
                misses.append(f"{f}: want {want}, got {got}")
        rows_ok += not misses
        detail = "; ".join(misses) or f"conf={pred.confidence:.2f}"
        print(f"{'FAIL' if misses else 'PASS'} {i + 1:2d}. {title:<70} {detail}")

    print()
    if error:
        print(f"Stopped early ({error}); {len(cases) - scored_rows} case(s) not scored.")
    if scored_rows:
        print(f"Overall: {rows_ok}/{scored_rows} ({rows_ok / scored_rows:.0%})")
    for f in FIELDS:
        if field_total[f]:
            print(f"  {f}: {field_hits[f]}/{field_total[f]} ({field_hits[f] / field_total[f]:.0%})")


if __name__ == "__main__":
    main()
