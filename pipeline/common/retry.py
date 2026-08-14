"""Shared retry policy for external API calls (Voyage, Anthropic, Cohere).

See docs/tehnicheskoe_zadanie.md, section 11: retry with exponential backoff
on transient errors only (429 rate limit, 5xx, connection timeouts) - not
4xx errors, which signal a bug in the request itself and are not fixed by
retrying.

Verified against the actual exception hierarchies of all three SDKs used in
this pipeline (voyageai 0.5.0, anthropic, cohere) on 2026-08-15: each
attaches a numeric status code to its API exceptions (`http_status` on
voyageai.error.VoyageError, `status_code` on anthropic.APIStatusError and
cohere's ApiError). Connection/timeout-level errors from all three SDKs
carry no status code and are identified by class name instead.
"""

from __future__ import annotations

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

STOP_AFTER_ATTEMPT = 5
WAIT_MIN_SECONDS = 1
WAIT_MAX_SECONDS = 60

# Class-name fragments that identify a connection/timeout-level error when no
# numeric status code is attached (APIConnectionError, APITimeoutError,
# ServiceUnavailableError, TryAgain, etc. across the three SDKs).
_TRANSIENT_NAME_FRAGMENTS = ("Connection", "Timeout", "ServiceUnavailable", "TryAgain", "Overloaded")


def _status_code(exc: BaseException) -> int | None:
    """Read the HTTP status code off an SDK exception, if present.

    voyageai uses `http_status`; anthropic and cohere use `status_code`.
    """
    for attr in ("status_code", "http_status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def is_transient_error(exc: BaseException) -> bool:
    """True for errors worth retrying: 429, 5xx, and connection/timeout
    failures. False for 4xx errors and anything else - retrying those only
    masks a real bug with useless repeated calls (spec section 11).
    """
    status = _status_code(exc)
    if status is not None:
        return status == 429 or 500 <= status < 600
    name = type(exc).__name__
    return any(fragment in name for fragment in _TRANSIENT_NAME_FRAGMENTS)


def retryable():
    """Decorator factory applying the project's standard retry policy.

    Usage: @retryable() on any function making a single external API call.
    """
    return retry(
        retry=retry_if_exception(is_transient_error),
        stop=stop_after_attempt(STOP_AFTER_ATTEMPT),
        wait=wait_random_exponential(min=WAIT_MIN_SECONDS, max=WAIT_MAX_SECONDS),
        reraise=True,
    )
