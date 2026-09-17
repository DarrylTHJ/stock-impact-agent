"""Turn a user's wording into transparent retrieval phrases.

This module does not decide sector impacts. It only normalises the event before
the Chen and HLIB knowledge bases are searched independently.
"""

import os

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field


class EventAnalysis(BaseModel):
    event_summary: str = Field(description="Short factual restatement of the event.")
    location: str | None = Field(default=None, description="Location only if stated or clear.")
    themes: list[str] = Field(default_factory=list, description="Relevant concepts, not impacts.")
    retrieval_queries: list[str] = Field(
        min_length=1,
        max_length=5,
        description="Short search phrases for the knowledge bases. Do not state impact direction.",
    )
    used_fallback: bool = False
    status_message: str | None = None


def analyse_event(event_text: str) -> EventAnalysis:
    """Use Gemini for query normalisation, with a safe non-LLM fallback."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY2")
    fallback = EventAnalysis(
        event_summary=event_text.strip(),
        retrieval_queries=[event_text.strip()],
        used_fallback=True,
        status_message="Gemini event analysis was unavailable; the original text was used for retrieval.",
    )
    if not api_key:
        return fallback

    prompt = f"""
You prepare retrieval queries for a Bursa Malaysia event-impact application.
Analyse the event below. Restate only what the event says; do not predict which
sector, company, or stock will benefit or suffer. Produce concise themes and
search phrases that help retrieve source evidence. Make the retrieval phrases
meaningfully different: cover the exact actors/policy, the economic mechanism,
the relevant Malaysian exposure, and any stated geopolitical or supply-chain
risk. Do not produce several generic variations of the same phrase.

Event: {event_text}
"""
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            # gemini-2.5-flash is retired for newly created projects. This
            # lightweight current model is sufficient for query normalisation.
            model="gemini-3.1-flash-lite",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": EventAnalysis,
                "temperature": 0,
            },
        )
        return EventAnalysis.model_validate_json(response.text)
    except Exception:
        # The app remains usable if the API key, network, or quota is unavailable.
        return fallback
