"""Convert user-provided text, an article URL, or a PDF into event text."""

from dataclasses import dataclass
from io import BytesIO
import json

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


MAX_EVENT_TEXT_LENGTH = 30_000


@dataclass
class EventSource:
    text: str
    label: str
    warning: str | None = None


def from_typed_text(text: str) -> EventSource:
    return EventSource(text=text.strip(), label="Typed event description")


def json_ld_article_body(soup: BeautifulSoup) -> str | None:
    """Return an articleBody value when a publisher exposes one in JSON-LD."""
    def find_body(value: object) -> str | None:
        if isinstance(value, dict):
            body = value.get("articleBody")
            if isinstance(body, str) and body.strip():
                return body.strip()
            for child in value.values():
                found = find_body(child)
                if found:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = find_body(child)
                if found:
                    return found
        return None

    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            found = find_body(json.loads(tag.get_text()))
            if found:
                return found
        except json.JSONDecodeError:
            continue
    return None


def readable_text(node: object) -> str:
    """Prefer prose blocks over menu labels and page chrome."""
    if not hasattr(node, "find_all"):
        return ""
    blocks = node.find_all(["h1", "h2", "p"])
    text = " ".join(block.get_text(" ", strip=True) for block in blocks)
    return text.strip()


def from_url(url: str) -> EventSource:
    if not url.startswith(("https://", "http://")):
        raise ValueError("Enter a complete URL beginning with https:// or http://.")

    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; FYP event analyser)"},
        timeout=15,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    structured_article = json_ld_article_body(soup)
    for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        element.decompose()
    for element in soup.find_all(attrs={"role": "navigation"}):
        element.decompose()
    for element in soup.find_all(True):
        # A parent may have been removed earlier in this loop, leaving one of
        # its descendant tags detached with attrs=None.
        attrs = element.attrs or {}
        labels = " ".join(attrs.get("class") or []) + " " + (attrs.get("id") or "")
        if any(token in labels.lower() for token in ("breadcrumb", "cookie", "menu", "navbar", "sidebar", "site-header", "site-footer")):
            element.decompose()

    candidates = soup.find_all("article")
    candidates.extend(soup.select("[itemprop='articleBody'], .article-body, .article-content, .story-body, .post-content"))
    candidates.extend(soup.find_all("main"))
    extracted = [readable_text(candidate) for candidate in candidates]
    text = structured_article or max(extracted, key=len, default="")
    if len(text) < 120:
        raise ValueError(
            "This URL did not expose enough readable article text. It may be a homepage, "
            "a navigation page, paywalled, or JavaScript-rendered. Paste the event text instead."
        )
    return EventSource(
        text=text[:MAX_EVENT_TEXT_LENGTH],
        label=f"Article URL: {url}",
        warning="Only readable webpage text was extracted; paywalled or JavaScript-only articles may not work.",
    )


def from_pdf(file_bytes: bytes, filename: str) -> EventSource:
    reader = PdfReader(BytesIO(file_bytes))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    text = "\n".join(page for page in pages if page)
    if not text:
        raise ValueError("No selectable text was found. This PDF may be scanned and require OCR.")
    return EventSource(
        text=text[:MAX_EVENT_TEXT_LENGTH],
        label=f"Uploaded PDF: {filename}",
        warning="Only the first 30,000 extracted characters are used for event analysis.",
    )
