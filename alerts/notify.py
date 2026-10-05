"""Send alert text to the Discord webhook. Never raises: a failed send is logged and
reported as False so the caller (and the pipeline) carries on."""

import logging

import httpx

from config import get_settings

logger = logging.getLogger(__name__)

TIMEOUT = 10.0
# Discord rejects message content longer than this.
MAX_LENGTH = 2000


def _truncate(text: str) -> str:
    return text if len(text) <= MAX_LENGTH else text[: MAX_LENGTH - 1] + "…"


def make_client() -> httpx.Client:
    return httpx.Client(timeout=TIMEOUT)


def send(text: str, *, url: str | None = None, client: httpx.Client | None = None) -> bool:
    """Post text to ALERT_WEBHOOK_URL (or `url`). With no URL configured the alert is
    only logged, which counts as sent. Returns False if the send failed."""
    url = url if url is not None else get_settings().alert_webhook_url
    if not url:
        logger.warning("ALERT (log only, ALERT_WEBHOOK_URL unset): %s", text)
        return True

    # parse=[] so text from article titles can't ping @everyone.
    payload = {"content": _truncate(text), "allowed_mentions": {"parse": []}}
    own_client = client is None
    try:
        if own_client:
            client = make_client()
        response = client.post(url, json=payload)
        if response.status_code >= 300:
            logger.error(
                "Alert webhook returned %d: %s", response.status_code, response.text[:200]
            )
            return False
        logger.info("Alert sent: %s", text.splitlines()[0] if text else "")
        return True
    except Exception as exc:
        logger.error("Alert webhook failed (%s: %s); alert was: %s", type(exc).__name__, exc, text)
        return False
    finally:
        if own_client and client is not None:
            client.close()
