"""Convert user-provided text, an article URL, or a PDF into event text."""

from dataclasses import dataclass
from io import BytesIO

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
    for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
        element.decompose()
    content = soup.find("article") or soup.find("main") or soup.body
    text = content.get_text(" ", strip=True) if content else ""
    if not text:
        raise ValueError("No readable article text was found at this URL.")
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
