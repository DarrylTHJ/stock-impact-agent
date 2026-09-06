import streamlit as st

from event_analysis import analyse_event
from event_input import from_pdf, from_typed_text, from_url
from knowledge_store import retrieve_records


st.set_page_config(page_title="Bursa Event Impact Explorer", layout="wide")
st.title("Bursa Event Impact Explorer")
st.caption("Evidence-grounded comparison of Chen commentary and HLIB research. Not investment advice.")

input_mode = st.radio(
    "Event input", ["Type event", "Paste article URL", "Upload PDF"], horizontal=True
)

event_text = ""
event_url = ""
uploaded_pdf = None
if input_mode == "Type event":
    event_text = st.text_area(
        "Describe an event in English",
        placeholder="Example: Malaysia announces new data-centre projects in Johor.",
        height=110,
    )
elif input_mode == "Paste article URL":
    event_url = st.text_input("Article URL", placeholder="https://example.com/news-article")
else:
    uploaded_pdf = st.file_uploader("Event PDF", type=["pdf"])


def render_source_column(source_name: str, search_queries: list[str]) -> None:
    st.subheader(source_name)
    records = retrieve_records(search_queries, source_name)
    if not records:
        st.info("No matching knowledge records found.")
        return

    for record in records:
        if record.validate_for_graph():
            direction = {"positive": "Positive", "negative": "Negative", "mixed": "Mixed"}[record.impact_direction]
            target = record.impacted_sector
            if record.impacted_industry:
                target = f"{record.impacted_industry} (within {record.impacted_sector})"
            st.markdown(f"### {target}: {direction}")
        elif record.validate_for_company_graph():
            direction = {"positive": "Positive", "negative": "Negative", "mixed": "Mixed"}[record.impact_direction]
            companies = ", ".join(company.company_name for company in record.impacted_companies)
            target = companies
            if record.impacted_industry:
                target = f"{companies} ({record.impacted_industry})"
            st.markdown(f"### {target}: {direction}")
        else:
            st.markdown("### Market context")

        st.write(record.reason)
        with st.expander("Evidence and source"):
            st.caption(f"{record.source_title} | {record.source_date}")
            for item in record.evidence:
                st.markdown(f"> {item.quote}")
                if item.translation:
                    st.caption(item.translation)
                st.caption(item.location)
            if record.source_link:
                st.link_button("Open original source", record.source_link)


has_input = bool(event_text.strip() or event_url.strip() or uploaded_pdf)
if st.button("Analyse event", type="primary", disabled=not has_input):
    try:
        if input_mode == "Type event":
            event_source = from_typed_text(event_text)
        elif input_mode == "Paste article URL":
            with st.spinner("Extracting readable article text..."):
                event_source = from_url(event_url.strip())
        else:
            with st.spinner("Extracting text from PDF..."):
                event_source = from_pdf(uploaded_pdf.getvalue(), uploaded_pdf.name)
    except Exception as error:
        st.error(f"Could not prepare the event input: {error}")
        st.stop()

    st.caption(event_source.label)
    if event_source.warning:
        st.info(event_source.warning)

    with st.spinner("Analysing the event wording..."):
        event_analysis = analyse_event(event_source.text)

    with st.expander("Event analysis used for retrieval", expanded=False):
        st.write(event_analysis.event_summary)
        if event_analysis.used_fallback:
            st.warning(event_analysis.status_message)
        else:
            st.success("Gemini normalised the event into retrieval phrases.")
        if event_analysis.location:
            st.caption(f"Location: {event_analysis.location}")
        st.caption("Retrieval phrases: " + " | ".join(event_analysis.retrieval_queries))

    chen_column, hlib_column = st.columns(2)
    with chen_column:
        render_source_column("Chen", event_analysis.retrieval_queries)
    with hlib_column:
        render_source_column("HLIB Research", event_analysis.retrieval_queries)
else:
    st.info("Provide an event as text, an article URL, or a PDF, then select Analyse event.")
