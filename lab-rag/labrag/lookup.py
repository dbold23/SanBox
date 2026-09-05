"""Fill in title / authors / year from the DOI using Crossref.

PDF metadata is unreliable and filenames are inconsistent; the DOI printed on
the first page is not. One GET to api.crossref.org per new paper gives a clean
citation. Works without a network too: on any failure we keep the heuristics.
"""

from __future__ import annotations

import logging

from .parse import ParsedDoc

log = logging.getLogger(__name__)

CROSSREF_URL = "https://api.crossref.org/works/{doi}"
USER_AGENT = "LabRAG/2.0 (https://github.com/dbold23/ocean-predator-ecology-lab; mailto:labrag@example.org)"


class CrossrefLookup:
    def __init__(self, timeout: float = 8.0, client=None):
        import httpx

        self._client = client or httpx.Client(timeout=timeout, headers={"User-Agent": USER_AGENT})
        self._httpx = httpx
        self.failures = 0
        self.max_failures = 5  # stop trying once the network is clearly down

    def enrich(self, doc: ParsedDoc) -> bool:
        """Overwrite title/authors/year on doc from Crossref. Returns True if it did."""
        if not doc.doi or self.failures >= self.max_failures:
            return False
        try:
            r = self._client.get(CROSSREF_URL.format(doi=doc.doi))
        except Exception as exc:  # network down, DNS, timeout
            self.failures += 1
            log.info("Crossref unreachable (%s); keeping heuristic metadata", exc)
            return False
        if r.status_code == 404:
            return False  # the regex picked up a bad DOI; nothing to do
        if r.status_code >= 400:
            self.failures += 1
            return False
        self.failures = 0
        try:
            message = r.json()["message"]
        except (ValueError, KeyError):
            return False
        return apply_crossref(doc, message)


def apply_crossref(doc: ParsedDoc, message: dict) -> bool:
    changed = False
    titles = message.get("title") or []
    if titles and titles[0].strip():
        doc.title = " ".join(titles[0].split())[:300]
        changed = True
    authors = message.get("author") or []
    names = []
    for a in authors:
        family = (a.get("family") or "").strip()
        given = (a.get("given") or "").strip()
        if family:
            names.append(f"{family} {given[:1]}".strip() if given else family)
        elif a.get("name"):
            names.append(a["name"].strip())
    if names:
        doc.authors = ", ".join(names[:12]) + (" et al." if len(names) > 12 else "")
        changed = True
    for key in ("published-print", "published-online", "issued", "created"):
        parts = (message.get(key) or {}).get("date-parts") or []
        if parts and parts[0] and parts[0][0]:
            doc.year = int(parts[0][0])
            changed = True
            break
    return changed
