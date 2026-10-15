"""Internet search and page reading for the chat (DuckDuckGo via the `ddgs` package; no API key).

Quick answers use search snippets as secondary facts (capped share). Research reports use the internet
as their primary evidence: several targeted searches (web and news), then the top pages are fetched
and the paragraphs most relevant to the question are kept, with the page's publication date when it
can be read. Only search queries and page requests leave the machine - never PDF text.

Every failure (no network, rate limiting, timeouts, unreadable pages) degrades to fewer results with a
note; nothing is invented to fill the gap.
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from html import unescape
from urllib.parse import urlparse

log = logging.getLogger(__name__)

# Sites that tend to add noise rather than facts for this domain.
_SKIP_DOMAINS = ("youtube.com", "facebook.com", "instagram.com", "x.com", "twitter.com", "pinterest.",
                 "tiktok.com", "quora.com")
# Domains preferred when ranking research sources.
_AUTHORITATIVE = ("gov.in", "nic.in", "coalindia.in", "easterncoal", "dgms", "coal.gov", "ibm.gov",
                  "cmpdi", "nih.gov", "sciencedirect", "springer", "researchgate", "wikipedia.org",
                  "pib.gov.in", "thehindu", "indianexpress", "timesofindia", "hindustantimes", "business-standard")
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
       "Chrome/124.0 Safari/537.36 Inundation-research")


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str
    date: str | None = None  # publication date if known (YYYY-MM-DD or as published)
    source: str = "web"  # web | news

    @property
    def domain(self) -> str:
        return urlparse(self.url).netloc.removeprefix("www.")

    @property
    def authority(self) -> int:
        return int(any(a in self.domain.lower() for a in _AUTHORITATIVE))


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _accept(url: str, seen: set[str]) -> bool:
    if not url.startswith(("https://", "http://")) or url in seen:
        return False
    return not any(d in urlparse(url).netloc.lower() for d in _SKIP_DOMAINS)


def search_web(query: str, max_results: int = 4, timeout_s: int = 7) -> tuple[list[WebResult], str | None]:
    """Return (results, error). Only http(s) links with a title and snippet are kept."""
    try:
        from ddgs import DDGS
    except ImportError:
        return [], "Web search is unavailable: the 'ddgs' package is not installed."
    try:
        # the DuckDuckGo backend directly: "auto" may route through slower engines that time out
        raw = DDGS(timeout=timeout_s).text(query, max_results=max_results + 3, safesearch="moderate",
                                           backend="duckduckgo") or []
    except Exception as exc:  # network errors, rate limits, parser changes
        if "no results" in str(exc).lower():
            return [], None  # an empty result set is not a failure
        log.warning("Web search failed (%s)", type(exc).__name__)
        return [], f"Web search failed ({type(exc).__name__}); the answer uses the PDFs only."
    results: list[WebResult] = []
    seen: set[str] = set()
    for r in raw:
        url = (r.get("href") or r.get("url") or "").strip()
        title, snippet = _clean(r.get("title", "")), _clean(r.get("body", ""))
        if not _accept(url, seen) or not title or len(snippet) < 30:
            continue
        seen.add(url)
        results.append(WebResult(title=title[:160], url=url, snippet=snippet[:700]))
        if len(results) >= max_results:
            break
    return results, None if results else "Web search returned no usable results."


def search_news(query: str, max_results: int = 4, timeout_s: int = 7) -> list[WebResult]:
    """News results carry publication dates, which matter for claims about current conditions."""
    try:
        from ddgs import DDGS

        raw = DDGS(timeout=timeout_s).news(query, max_results=max_results + 2, region="in-en") or []
    except Exception as exc:
        log.warning("News search failed (%s)", type(exc).__name__)
        return []
    out, seen = [], set()
    for r in raw:
        url = (r.get("url") or r.get("href") or "").strip()
        title, body = _clean(r.get("title", "")), _clean(r.get("body", ""))
        if not _accept(url, seen) or not title or len(body) < 30:
            continue
        seen.add(url)
        out.append(WebResult(title=title[:160], url=url, snippet=body[:700],
                             date=(r.get("date") or "")[:10] or None, source="news"))
        if len(out) >= max_results:
            break
    return out


def _wikipedia(topic: str, timeout_s: int) -> list[WebResult]:
    """Fallback when general search returns almost nothing: the topic's Wikipedia article."""
    try:
        from ddgs import DDGS

        raw = DDGS(timeout=timeout_s).text(topic, max_results=2, backend="wikipedia") or []
    except Exception:
        return []
    return [WebResult(title=_clean(r.get("title", ""))[:160], url=r.get("href", ""), snippet=_clean(r.get("body", ""))[:700])
            for r in raw if (r.get("href") or "").startswith("https://")]


# --------------------------------------------------------------------------- page reading
_DATE_META = re.compile(
    r"<meta[^>]+(?:property|name|itemprop)=[\"'](?:article:published_time|og:published_time|datePublished|"
    r"publish[_-]?date|date|dc\.date|DC\.date\.issued|pubdate)[\"'][^>]*content=[\"']([^\"']{8,40})", re.I)
_DATE_JSON = re.compile(r"\"datePublished\"\s*:\s*\"([^\"]{8,40})\"")


def _html_to_paragraphs(html: str) -> list[str]:
    html = re.sub(r"(?is)<(script|style|noscript|svg|nav|footer|header|form|aside)[^>]*>.*?</\1>", " ", html)
    html = re.sub(r"(?i)<br\s*/?>", "\n", html)
    html = re.sub(r"(?i)</(p|div|li|h[1-6]|tr|section|article|td)>", "\n", html)
    text = unescape(re.sub(r"<[^>]+>", " ", html))
    return [p for p in (_clean(x) for x in text.split("\n")) if len(p) >= 60]


def _pdf_to_paragraphs(data: bytes, max_pages: int = 15) -> list[str]:
    import pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as doc:
        text = "\n".join(doc[i].get_text("text") for i in range(min(max_pages, doc.page_count)))
    paras = re.split(r"\n\s*\n|(?<=\.)\n", text)
    return [p for p in (_clean(x) for x in paras) if len(p) >= 60]


def _best_paragraphs(paragraphs: list[str], terms: list[str], max_chars: int) -> str:
    """Keep the paragraphs that mention the most query terms, in their original order."""
    terms = [t.lower() for t in terms if len(t) >= 3]
    scored = [(sum(p.lower().count(t) for t in terms), i, p) for i, p in enumerate(paragraphs)]
    chosen = sorted([x for x in scored if x[0] > 0], key=lambda x: -x[0])
    keep, total = [], 0
    for _score, i, p in chosen:
        if total + len(p) > max_chars:
            continue
        keep.append((i, p))
        total += len(p)
    return " ... ".join(p for _, p in sorted(keep))


def fetch_page(url: str, terms: list[str], timeout_s: int = 8, max_chars: int = 1600) -> tuple[str, str | None]:
    """Fetch one page (HTML or PDF) and return (relevant excerpt, publication date or None)."""
    import httpx

    with httpx.Client(timeout=timeout_s, follow_redirects=True, headers={"User-Agent": _UA}) as client:
        resp = client.get(url)
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "").lower()
    data = resp.content[:6_000_000]
    if "pdf" in ctype or url.lower().endswith(".pdf"):
        return _best_paragraphs(_pdf_to_paragraphs(data), terms, max_chars), None
    html = data.decode(resp.encoding or "utf-8", errors="replace")
    m = _DATE_META.search(html) or _DATE_JSON.search(html)
    return _best_paragraphs(_html_to_paragraphs(html), terms, max_chars), (m.group(1)[:10] if m else None)


def research(queries: list[str], terms: list[str], *, per_query: int = 4, max_sources: int = 8,
             fetch: int = 5, timeout_s: int = 8, must_mention: list[str] | None = None) -> tuple[list[WebResult], list[str]]:
    """Run several searches in parallel, rank and de-duplicate, then read the top pages.

    Returns (sources, notes). A fetched page replaces its search snippet with the most relevant
    paragraphs; if fetching fails, the snippet is kept.
    """
    import time

    notes: list[str] = []
    jobs = [(search_web, q) for q in queries] + [(search_news, q) for q in queries[:1]]
    batches = []
    # one search at a time with a short pause: the free search endpoint throttles rapid bursts
    for n, (fn, q) in enumerate(jobs):
        if n:
            time.sleep(0.6)
        try:
            res = fn(q, per_query, timeout_s)
        except Exception as exc:  # defensive: a failed search must not stop the others
            notes.append(f"A search failed ({type(exc).__name__}).")
            continue
        if isinstance(res, tuple):
            res, err = res
            if err and not res:
                notes.append(err)
        batches.append(res)
    if sum(len(b) for b in batches) < 2 and must_mention:
        batches.append(_wikipedia(must_mention[0], timeout_s))
    merged: dict[str, WebResult] = {}
    for rank in range(max(len(b) for b in batches) if batches else 0):  # interleave so every query is heard
        for batch in batches:
            if rank < len(batch) and batch[rank].url not in merged:
                merged[batch[rank].url] = batch[rank]
    if must_mention:
        keys = [k.lower() for k in must_mention if len(k) >= 3]
        squeezed = [k.replace(" ", "") for k in keys]
        dropped = [r for r in merged.values()
                   if not any(k in f"{r.title} {r.snippet}".lower() for k in keys)
                   and not any(k in r.url.lower().replace("-", "").replace("_", "") for k in squeezed)]
        for r in dropped:
            merged.pop(r.url, None)
        if dropped:
            notes.append(f"{len(dropped)} search result(s) were ignored as unrelated to the question.")
    ranked = sorted(merged.values(), key=lambda r: -r.authority)[:max_sources]
    notes = list(dict.fromkeys(notes))
    if not ranked:
        notes.append("Internet research returned no usable sources.")
        return [], notes

    def read(r: WebResult) -> WebResult:
        try:
            text, date = fetch_page(r.url, terms, timeout_s)
            if len(text) >= 200:
                r.snippet = text
            r.date = r.date or date
        except Exception as exc:  # unreachable, blocked or unreadable page: keep the snippet
            log.info("Could not read %s (%s)", r.domain, type(exc).__name__)
        return r

    with ThreadPoolExecutor(max_workers=fetch) as pool:
        ranked[:fetch] = list(pool.map(read, ranked[:fetch]))
    return ranked, notes
