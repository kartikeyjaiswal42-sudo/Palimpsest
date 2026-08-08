"""Fetchers for every corpus source, one function per source shape.

Each returns a list of :class:`Document`. Nothing here writes to the repository -- the
caller decides where the corpus lands, and ``.gitignore`` keeps the human text out of git.

A note on HTTP: this module uses ``httpx``, which ships its own CA bundle via ``certifi``.
The system Python on macOS frequently has no usable CA store, and ``urllib`` fails every
one of these URLs with CERTIFICATE_VERIFY_FAILED. If you rewrite this with ``urllib``, it
will appear to be a network outage and is not one.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import parse_qsl

import httpx

from .sources import GHOSTBUSTER_BASE, LIANG_BASE, Source

log = logging.getLogger(__name__)

__all__ = ["Document", "fetch_source", "write_jsonl", "read_jsonl", "USER_AGENT"]

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
# JHU's WordPress install rejects short user-agent strings with a 403 on some endpoints.
# It answers the same request happily with a full browser UA. Not rate limiting -- a WAF rule.

_CHATGPT_LAUNCH = "2022-11-30"


@dataclass
class Document:
    """One essay, with everything needed to trace it back to where it came from."""

    id: str
    source_id: str
    text: str
    #: "human" | "machine" | "hybrid"
    authorship: str
    role: str
    n_words: int = 0
    sha256: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.text = _normalise(self.text)
        self.n_words = len(self.text.split())
        self.sha256 = hashlib.sha256(self.text.encode("utf-8")).hexdigest()


# Typography that differs by COLLECTION METHOD rather than by author. Web pages render
# smart quotes; an API returns ASCII. Mapped to a single canonical form everywhere.
_TYPOGRAPHY = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",   # single quotes
    "“": '"', "”": '"', "„": '"', "‟": '"',   # double quotes
    "′": "'", "″": '"',                                   # primes
    "–": " - ", "—": " - ", "―": " - ",             # en/em/horizontal dash
    "…": "...",                                                # ellipsis
    " ": " ", " ": " ", " ": " ", " ": " ",    # exotic spaces
    "​": "", "﻿": "",                                     # zero-width
}
_TYPO_RE = re.compile("|".join(map(re.escape, _TYPOGRAPHY)))


def _normalise(text: str) -> str:
    """Canonicalise whitespace and typography without touching the words.

    Whitespace normalisation is obvious housekeeping. The typography mapping is not, and it
    is load-bearing.

    Our human essays were scraped from college web pages, which render smart quotes; our
    machine essays arrived through an API as plain ASCII. Measured across the two halves of
    the training corpus BEFORE this function existed:

        curly apostrophe   88.1% of human docs   9.7% of machine docs
        curly quotes       72.3%                 0.0%
        em dash            43.6%                 0.0%

    A detector trained on that is a smart-quote detector. Worse, the effect reaches the
    likelihood features too, because GPT-2 assigns a different token -- and therefore a
    different surprisal -- to U+2019 than to an ASCII apostrophe. Mapping both halves onto
    one convention removes an artefact of how the text was collected while leaving every
    word, and the presence of a dash or a quotation, intact.

    Digits are deliberately NOT touched. Their near-absence from the machine half is a
    property of the prose, not of the pipeline.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TYPO_RE.sub(lambda m: _TYPOGRAPHY[m.group(0)], text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _client() -> httpx.Client:
    return httpx.Client(
        timeout=120,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json, text/html, */*"},
    )


# --------------------------------------------------------------------------- fetchers


def fetch_liang_json(source: Source, client: httpx.Client, limit: int | None = None) -> list[Document]:
    """Liang et al. ship each set as a JSON array of ``{document, ...}``."""
    r = client.get(source.url)
    r.raise_for_status()
    rows = r.json()
    docs = []
    for i, row in enumerate(rows[: limit or len(rows)]):
        text = row.get("document", "")
        if not text.strip():
            continue
        meta = {k: v for k, v in row.items() if k != "document"}
        docs.append(
            Document(
                id=f"{source.id}:{i:04d}",
                source_id=source.id,
                text=text,
                authorship=source.authorship,
                role=source.role,
                meta=meta,
            )
        )
    return docs


def fetch_ghostbuster_txt(
    source: Source, client: httpx.Client, limit: int | None = None
) -> list[Document]:
    """1,000 flat .txt files. Fetched individually -- the full repo is 651 MB, do not clone."""
    n = limit or source.expected_n
    docs = []
    for i in range(1, n + 1):
        try:
            r = client.get(f"{GHOSTBUSTER_BASE}/{i}.txt")
            if r.status_code != 200 or len(r.text.split()) < 50:
                continue  # the set contains at least one empty file
            docs.append(
                Document(
                    id=f"{source.id}:{i:04d}",
                    source_id=source.id,
                    text=r.text,
                    authorship=source.authorship,
                    role=source.role,
                )
            )
        except httpx.HTTPError as exc:
            log.warning("ghostbuster %d failed: %s", i, exc)
        if i % 100 == 0:
            log.info("ghostbuster %d/%d", i, n)
    return docs


def fetch_hamilton_html(
    source: Source, client: httpx.Client, limit: int | None = None
) -> list[Document]:
    """Hamilton publishes four year-archives as HTML.

    Structure verified by inspection: each essay is an ``<h3>`` (author name), then an
    ``<h5>`` (hometown), then the body paragraphs. Splitting on the wrapping ``<div>``
    instead merges several essays into one -- some years put multiple essays in one
    container -- so we split on ``<h3>`` and sanity-check the word count.
    """
    from bs4 import BeautifulSoup

    docs = []
    for year in source.extra.get("years", []):
        url = f"{source.url}{year}-essays-that-worked"
        r = client.get(url)
        if r.status_code != 200:
            log.warning("hamilton %s -> %s", year, r.status_code)
            continue
        soup = BeautifulSoup(r.text, "lxml")
        main = soup.find("main") or soup
        for h3 in main.find_all("h3"):
            author = h3.get_text(strip=True)
            hometown = ""
            paras: list[str] = []
            for sib in h3.find_next_siblings():
                if sib.name == "h3":
                    break
                if sib.name == "h5" and not hometown:
                    hometown = sib.get_text(strip=True)
                elif sib.name == "p":
                    t = sib.get_text(" ", strip=True)
                    if t:
                        paras.append(t)
            text = "\n\n".join(paras)
            words = len(text.split())
            # A merge artefact shows up as an implausibly long "essay". The Common App cap
            # is 650 words; anything past 1,500 is two essays glued together.
            if words < 150 or words > 1500:
                log.warning("hamilton %s/%s skipped: %d words", year, author, words)
                continue
            docs.append(
                Document(
                    id=f"{source.id}:{year}:{_slug(author)}",
                    source_id=source.id,
                    text=text,
                    authorship=source.authorship,
                    role=source.role,
                    meta={"year": year, "author": author, "hometown": hometown},
                )
            )
    return docs


def fetch_jhu_wp_api(
    source: Source, client: httpx.Client, limit: int | None = None
) -> list[Document]:
    """Johns Hopkins, via the open WordPress REST API.

    Two traps, both verified the hard way:

    1. The ``insider_article`` post type holds all 443 blog posts, not just essays. We
       narrow it with the ``essays-that-worked-*`` tag ids, discovered at runtime.
    2. Those tag names encode the GRADUATING CLASS, not the publication year -- the tag
       ``essays-that-worked-2020`` returns posts published 2016-12-15. Filtering on the tag
       name to get pre-ChatGPT essays would silently mislabel the corpus. We filter on the
       API's own ``date`` field instead.
    """
    from bs4 import BeautifulSoup

    tags = client.get(
        "https://apply.jhu.edu/wp-json/wp/v2/tags",
        params={"per_page": 100, "search": "essays-that-worked", "_fields": "id,slug"},
    )
    tags.raise_for_status()
    tag_ids = ",".join(str(t["id"]) for t in tags.json())
    if not tag_ids:
        return []

    posts = client.get(
        "https://apply.jhu.edu/wp-json/wp/v2/insider_article",
        params={"tags": tag_ids, "per_page": 100, "_fields": "slug,date,title,content"},
    )
    posts.raise_for_status()

    docs = []
    for post in posts.json():
        date = post.get("date", "")
        if date[:10] > _CHATGPT_LAUNCH:
            continue  # keep only provably pre-ChatGPT publications
        html = (post.get("content") or {}).get("rendered", "")
        soup = BeautifulSoup(html, "lxml")
        paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = "\n\n".join(p for p in paras if p)
        if len(text.split()) < 150:
            continue
        docs.append(
            Document(
                id=f"{source.id}:{post['slug'][:60]}",
                source_id=source.id,
                text=text,
                authorship=source.authorship,
                role=source.role,
                meta={
                    "date": date[:10],
                    "title": (post.get("title") or {}).get("rendered", "")[:120],
                },
            )
        )
    return docs


def fetch_ellipse_csv(
    source: Source, client: httpx.Client, limit: int | None = None
) -> list[Document]:
    """ELLIPSE: one CSV, every writer an English Language Learner, with graded proficiency.

    The ``Overall`` column is a 1.0-5.0 holistic proficiency score. That is strictly more
    useful than a binary ESL flag: it lets us ask whether the detector penalises *weaker*
    English specifically, rather than only whether it penalises non-native writers at all.
    """
    import csv
    import io

    r = client.get(source.url)
    r.raise_for_status()
    reader = csv.DictReader(io.StringIO(r.text))
    text_col = source.extra.get("text_column", "full_text")
    prof_col = source.extra.get("proficiency_column", "Overall")

    docs = []
    for i, row in enumerate(reader):
        if limit and len(docs) >= limit:
            break
        text = (row.get(text_col) or "").strip()
        if len(text.split()) < 100:
            continue
        try:
            proficiency = float(row.get(prof_col) or "nan")
        except ValueError:
            proficiency = float("nan")
        docs.append(
            Document(
                id=f"{source.id}:{i:05d}",
                source_id=source.id,
                text=text,
                authorship=source.authorship,
                role=source.role,
                meta={"proficiency": proficiency, "ell": True},
            )
        )
    return docs


def fetch_persuade_hf_rows(
    source: Source, client: httpx.Client, limit: int | None = None
) -> list[Document]:
    """PERSUADE 2.0 via the HuggingFace no-auth rows API.

    The page size maximum is 100 (``length=200`` returns 422). We deliberately keep both
    ELL and non-ELL rows: the value of this source is that it supplies a *matched* control
    -- same prompts, same graders, same cohort, differing only in the ELL flag.
    """
    ell_col = source.extra.get("ell_column", "ell_status")
    text_col = source.extra.get("text_column", "full_text")
    target = limit or source.expected_n

    # Two traps here, both of which fail silently as "0 documents fetched":
    #   1. httpx REPLACES a URL's existing query string when `params` is supplied; it does
    #      not merge. Passing {"offset": ...} against a URL carrying ?dataset=... drops the
    #      dataset and the API answers 422.
    #   2. The registry URL is percent-encoded (nlpatunt%2FD_persuade_2). Splitting the
    #      query by hand and handing the raw value back to httpx encodes the '%' again,
    #      producing %252F and a dataset that does not exist. parse_qsl decodes properly.
    base_url, _, query = source.url.partition("?")
    base_params = dict(parse_qsl(query))

    docs: list[Document] = []
    seen: set[str] = set()
    offset = 0
    while len(docs) < target and offset < 20000:
        r = client.get(base_url, params={**base_params, "offset": offset, "length": 100})
        if r.status_code != 200:
            log.warning("persuade offset %d -> %s", offset, r.status_code)
            break
        rows = r.json().get("rows", [])
        if not rows:
            break
        for item in rows:
            row = item.get("row", {})
            text = (row.get(text_col) or "").strip()
            if len(text.split()) < 100:
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen:
                continue  # the corpus is exploded one row per discourse element
            seen.add(digest)
            ell = str(row.get(ell_col) or "").strip()
            if ell not in ("Yes", "No"):
                continue
            docs.append(
                Document(
                    id=f"{source.id}:{digest[:12]}",
                    source_id=source.id,
                    text=text,
                    authorship=source.authorship,
                    role=source.role,
                    meta={
                        "ell": ell == "Yes",
                        "grade_level": row.get("grade_level"),
                        "prompt_name": row.get("prompt_name"),
                    },
                )
            )
            if len(docs) >= target:
                break
        offset += 100
    return docs


_FETCHERS = {
    "liang_json": fetch_liang_json,
    "ghostbuster_txt": fetch_ghostbuster_txt,
    "hamilton_html": fetch_hamilton_html,
    "jhu_wp_api": fetch_jhu_wp_api,
    "ellipse_csv": fetch_ellipse_csv,
    "persuade_hf_rows": fetch_persuade_hf_rows,
}


def fetch_source(source: Source, limit: int | None = None) -> list[Document]:
    """Dispatch to the right fetcher for ``source``."""
    fn = _FETCHERS.get(source.fetcher)
    if fn is None:
        raise ValueError(f"no fetcher named {source.fetcher!r}")
    with _client() as client:
        return fn(source, client, limit)


# --------------------------------------------------------------------------- storage


def write_jsonl(docs: list[Document], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[Document]:
    docs = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                docs.append(Document(**json.loads(line)))
    return docs


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40]
