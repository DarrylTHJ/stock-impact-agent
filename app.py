import streamlit as st

from event_analysis import analyse_event
from knowledge_store import retrieve_records


st.set_page_config(page_title="Bursa Event Impact Explorer", layout="wide")
st.title("Bursa Event Impact Explorer")
st.caption("Evidence-grounded comparison of Chen commentary and HLIB research. Not investment advice.")

event_text = st.text_area(
    "Describe an event in English",
    placeholder="Example: Malaysia announces new data-centre projects in Johor.",
    height=110,
)


def render_source_column(source_name: str, search_queries: list[str]) -> None:
    st.subheader(source_name)
    records = retrieve_records(search_queries, source_name)
    if not records:
        st.info("No matching knowledge records found.")
        return

    for record in records:
        if record.validate_for_graph():
            direction = {"positive": "Positive", "negative": "Negative", "mixed": "Mixed"}[record.impact_direction]
            st.markdown(f"### {record.impacted_sector}: {direction}")
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


if st.button("Analyse event", type="primary", disabled=not event_text.strip()):
    with st.spinner("Analysing the event wording..."):
        event_analysis = analyse_event(event_text)

    with st.expander("Event analysis used for retrieval", expanded=False):
        st.write(event_analysis.event_summary)
        if event_analysis.location:
            st.caption(f"Location: {event_analysis.location}")
        st.caption("Retrieval phrases: " + " | ".join(event_analysis.retrieval_queries))

    chen_column, hlib_column = st.columns(2)
    with chen_column:
        render_source_column("Chen", event_analysis.retrieval_queries)
    with hlib_column:
        render_source_column("HLIB Research", event_analysis.retrieval_queries)
else:
    st.info("Enter an event and select Analyse event to compare both evidence sources.")
