"""Score the real classifier against hand-labeled headlines: `python -m classifier.eval`.

Calls the live LLM in batches: the chain's first model by default, or one picked with
--provider provider:model (--model NAME is short for --provider gemini:NAME). Results
are cached in .eval_cache.json so reruns don't spend quota. Run manually; pytest does
not collect it.
"""

import argparse
import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import yaml

from classifier.classify import (
    BATCH_SIZE,
    LLMUnavailable,
    QuotaExhausted,
    classify_batch,
    take_batch,
)
from classifier.providers import ChainEntry, build_entry, chain_specs, parse_chain
from classifier.prefilter import prefilter
from classifier.prompt import PROMPT_VERSION
from classifier.schema import ClassifierOutput
from config import get_settings
from models import RawItem
from timeutil import to_local, to_utc

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "tests" / "fixtures" / "eval_headlines.yaml"
DEFAULT_CACHE = ROOT / ".eval_cache.json"
# Every field but the first three is only scored on cases that label it.
# creates_incident and occurred_at_null are derived from the prediction (see predicted).
FIELDS = (
    "is_crime", "crime_type", "in_stl", "was_shooting", "is_followup",
    "creates_incident", "occurred_at_null",
)


def predicted(pred: ClassifierOutput, field: str):
    if field == "creates_incident":
        # The matcher only creates incidents from St. Louis crimes that aren't follow-ups.
        return pred.is_crime and pred.in_stl and not pred.is_followup
    if field == "occurred_at_null":
        return pred.occurred_at is None
    return getattr(pred, field)


def load_cases(path: Path) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_cache(path: Path) -> dict[str, dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def save_cache(path: Path, cache: dict[str, dict]) -> None:
    path.write_text(json.dumps(cache, indent=1, sort_keys=True), encoding="utf-8")


def cache_model(model: str) -> str:
    """Gemini models keep their bare name, so results cached before models were
    named provider:model still count."""
    return model.removeprefix("gemini:")


def cache_key(case: dict, model: str) -> str:
    """Changes whenever the model, prompt version, or case text changes."""
    parts = [
        cache_model(model), PROMPT_VERSION, case["title"], case.get("body") or "", case.get("published_at") or ""
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def to_item(case: dict, item_id: int) -> RawItem:
    published = case.get("published_at")
    return RawItem(
        id=item_id,
        title=case["title"],
        body=case.get("body"),
        published_at=to_utc(datetime.fromisoformat(published)) if published else None,
    )


def predict_all(
    cases: list[dict], entry: ChainEntry, cache: dict[str, dict], use_cache: bool,
    sleep=time.sleep,
) -> tuple[dict[int, ClassifierOutput], str | None]:
    """Returns ({case index: prediction}, error message if the run stopped early).
    Cases that fail the prefilter are predicted not-crime without the LLM; new LLM
    results are written into `cache`."""
    llm = entry.client
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

    start = 0
    while start < len(pending):
        batch = take_batch(pending, start, entry.batch_size(BATCH_SIZE), entry.max_batch_tokens)
        start += len(batch)
        try:
            results = classify_batch(batch, llm, entry.limiter, sleep)
        except QuotaExhausted as exc:
            return preds, f"quota exhausted: {exc}"
        except LLMUnavailable as exc:
            return preds, f"LLM unavailable: {exc}"
        print(format_usage(len(batch), getattr(llm, "last_usage", None)))
        for item in batch:
            if item.id in results:
                fields = results[item.id].model_dump(exclude={"raw_item_id"})
                out = ClassifierOutput.model_validate(fields)
                preds[item.id] = out
                cache[cache_key(cases[item.id], llm.model)] = out.model_dump(mode="json")
    return preds, None


def format_usage(n_items: int, usage) -> str:
    """One line of token usage for a batch (the last request, if it was retried)."""
    if usage is None or usage.total_tokens is None:
        return f"Batch of {n_items}: no token usage reported"
    reasoning = f" (reasoning {usage.reasoning_tokens})" if usage.reasoning_tokens else ""
    return (
        f"Batch of {n_items}: prompt {usage.prompt_tokens}, completion "
        f"{usage.completion_tokens}{reasoning}, total {usage.total_tokens}"
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    which = parser.add_mutually_exclusive_group()
    which.add_argument(
        "--provider", metavar="PROVIDER:MODEL",
        help="model to evaluate instead of the chain's first, e.g. groq:openai/gpt-oss-120b",
    )
    which.add_argument("--model", help="short for --provider gemini:MODEL")
    parser.add_argument(
        "--no-cache", action="store_true",
        help="ignore cached results (fresh results are still saved)",
    )
    args = parser.parse_args(argv)

    cases = load_cases(args.path)
    settings = get_settings()
    if args.provider:
        try:
            specs = parse_chain(args.provider)
        except ValueError as exc:
            parser.error(str(exc))
        if len(specs) != 1:
            parser.error("--provider takes one provider:model")
        [(provider, model)] = specs
    elif args.model:
        provider, model = "gemini", args.model
    else:
        provider, model = chain_specs(settings)[0]
    entry = build_entry(provider, model, settings)
    print(f"Model: {entry.client.model} (prompt {PROMPT_VERSION})")
    cache = load_cache(args.cache)
    preds, error = predict_all(cases, entry, cache, use_cache=not args.no_cache)
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
            if f not in expected or (f == "in_stl" and want is None):
                continue
            field_total[f] += 1
            got = predicted(pred, f)
            if got == want:
                field_hits[f] += 1
            else:
                misses.append(f"{f}: want {want}, got {got}")
        rows_ok += not misses
        occurred = (
            to_local(pred.occurred_at).strftime("%Y-%m-%d %H:%M %Z") if pred.occurred_at else "-"
        )
        detail = "; ".join(misses) or f"conf={pred.confidence:.2f} occurred={occurred}"
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
