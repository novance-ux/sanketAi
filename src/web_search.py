"""Internet search for the chat's secondary facts (DuckDuckGo via the `ddgs` package; no API key).

The SANKET PDFs remain the primary source. Web results are a separate, clearly labelled set of passages
(W1, W2, ...) that may supply at most the configured share of an answer (default 20%), each fact cited
with its page link. Only the search query leaves the machine - never PDF text.

Failures (no network, rate limiting, timeouts) return no results and a note; the chat then answers from
the PDFs alone.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from urllib.parse import urlparse

log = logging.getLogger(__name__)

# Sites that tend to add noise rather than facts for this domain.
_SKIP_DOMAINS = ("youtube.com", "facebook.com", "instagram.com", "x.com", "twitter.com", "pinterest.")


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str

    @property
    def domain(self) -> str:
        return urlparse(self.url).netloc.removeprefix("www.")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def search_web(query: str, max_results: int = 4, timeout_s: int = 7) -> tuple[list[WebResult], str | None]:
    """Return (results, error). Only http(s) links with a title and snippet are kept."""
    try:
        from ddgs import DDGS
    except ImportError:
        return [], "Web search is unavailable: the 'ddgs' package is not installed."
    try:
        raw = DDGS(timeout=timeout_s).text(query, max_results=max_results + 3, safesearch="moderate") or []
    except Exception as exc:  # network errors, rate limits, parser changes
        log.warning("Web search failed (%s)", type(exc).__name__)
        return [], f"Web search failed ({type(exc).__name__}); the answer uses the PDFs only."
    results: list[WebResult] = []
    seen: set[str] = set()
    for r in raw:
        url = (r.get("href") or r.get("url") or "").strip()
        title, snippet = _clean(r.get("title", "")), _clean(r.get("body", ""))
        if not url.startswith(("https://", "http://")) or not title or len(snippet) < 30:
            continue
        domain = urlparse(url).netloc.lower()
        if any(d in domain for d in _SKIP_DOMAINS) or url in seen:
            continue
        seen.add(url)
        results.append(WebResult(title=title[:160], url=url, snippet=snippet[:700]))
        if len(results) >= max_results:
            break
    return results, None if results else "Web search returned no usable results."
