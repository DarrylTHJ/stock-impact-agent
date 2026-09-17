"""Single-page Streamlit interface for the Stock Investment Event-Impact Agent."""
import hashlib
from html import escape
from pathlib import Path

import streamlit as st

from comparison_analysis import build_graph_data, compare_source_results
from event_analysis import analyse_event
from event_input import from_pdf, from_typed_text, from_url
from impact_synthesis import synthesize_impacts
from interactive_graph import render_interactive_graph
from knowledge_store import retrieve_diverse_candidates
from relevance_filter import review_relevance
from source_context import format_seconds, load_direct_chen_transcripts, timestamped_video_url

ROOT = Path(__file__).parent
HLIB_SOURCE_DIR = (ROOT / "data" / "hlib_source").resolve()
STAGES = [
    ("Analyse query", "Understand the user query."),
    ("Retrieve knowledge", "Search related knowledge from Chen and HLIB independently."),
    ("Determine relevance", "Filter and classify every retrieved record."),
    ("Retrieve source context", "Load extra context from highly relevant sources."),
    ("Generate verdicts", "Build separate Chen and HLIB conclusions."),
    ("Compare and map", "Compare both agents in an interactive causal graph."),
]

st.set_page_config(page_title="Stock Investment Event-Impact Agent", page_icon="📈", layout="wide")
st.markdown("""
<style>
.stApp{background:#f4f7fb;color:#182230}.block-container{max-width:1500px;padding-top:.7rem}
[data-testid=stHeader],#MainMenu,footer{visibility:hidden}.st-key-top_header{position:sticky;top:.3rem;z-index:999;background:#f4f7fbf5;border:1px solid #d7e0ec;border-radius:16px;padding:.7rem 1rem;box-shadow:0 5px 18px #192a4614}
.brand{display:flex;gap:.75rem;align-items:center}.mark{display:grid;place-items:center;width:40px;height:40px;border-radius:12px;background:linear-gradient(135deg,#3768ee,#7a49e8);color:white;font-weight:800}.title{font-size:1.35rem;font-weight:760}.sub{font-size:.84rem;color:#637083}
.nav{display:flex;gap:.35rem;overflow:auto;border-top:1px solid #d8e0eb;margin-top:.5rem;padding-top:.5rem}.nav a{text-decoration:none;white-space:nowrap;color:#42526a;background:white;border:1px solid #d7e0ec;border-radius:99px;padding:.3rem .55rem;font-size:.7rem}.anchor{scroll-margin-top:180px}
.flow{display:grid;grid-template-columns:repeat(6,1fr);gap:.55rem}.flow>div,.panel,.impact{background:white;border:1px solid #d7e0ec;border-radius:13px;padding:.8rem}.num{display:grid;place-items:center;width:24px;height:24px;border-radius:50%;background:#506fe5;color:white;font-size:.7rem}.ft{font-weight:700;font-size:.8rem;margin:.4rem 0}.fd{font-size:.7rem;color:#69778a}.event{background:white;border:1px solid #d7e0ec;border-radius:14px;padding:1rem}.eyebrow{font-size:.68rem;letter-spacing:.08em;color:#365edc;font-weight:750}.summary{font-size:1.02rem;font-weight:620}.chip,.pill,.badge{display:inline-block;border-radius:99px;padding:.2rem .45rem;margin:.35rem .15rem 0 0;font-size:.68rem}.chip{background:#e9efff;color:#3451a8}.pill{border:1px solid #d4dce7;color:#687588}.badge.direct{background:#e8f5ee;color:#147a4b}.badge.inference{background:#fff3d7;color:#8a5b00}.source{font-size:1.15rem;font-weight:750}.kind,.context{color:#687588;font-size:.78rem}.impact{margin:.6rem 0}.positive{color:#147a4b}.negative{color:#c94141}.mixed{color:#976000}.empty{border:1px dashed #c3cedc;border-radius:10px;padding:.8rem;color:#6d798a}.quote{background:#f8fafc;padding:.65rem;border-radius:8px}.compare{background:white;border:1px solid #d7e0ec;border-radius:13px;padding:1rem}.disclaimer{text-align:center;color:#7a8797;font-size:.72rem;margin:1rem}
@media(max-width:1000px){.flow{grid-template-columns:repeat(3,1fr)}.st-key-top_header{position:relative}}
</style>""", unsafe_allow_html=True)

def anchor(n): st.markdown(f'<div id="stage-{n}" class="anchor"></div>', unsafe_allow_html=True)
def nav(ph): ph.markdown('<div class="nav">'+''.join(f'<a href="#stage-{i}">{i}. {escape(t)}</a>' for i,(t,_) in enumerate(STAGES,1))+'</div>', unsafe_allow_html=True)
def flow():
    st.markdown('<div class="flow">'+''.join(f'<div><span class="num">{i}</span><div class="ft">{escape(t)}</div><div class="fd">{escape(d)}</div></div>' for i,(t,d) in enumerate(STAGES,1))+'</div>', unsafe_allow_html=True)

def event_view(a):
    chips = ([f"📍 {a.location}"] if a.location else []) + a.themes
    st.markdown(f'<div class="event"><div class="eyebrow">EVENT UNDERSTOOD BY THE SYSTEM</div><div class="summary">{escape(a.event_summary)}</div>'+''.join(f'<span class="chip">{escape(x)}</span>' for x in chips)+'</div>', unsafe_allow_html=True)
    st.caption("Retrieval phrases: " + " · ".join(a.retrieval_queries))

def relevance_rows(review):
    return [{"Source":x.candidate.record.source_name,"Knowledge ID":x.candidate.record.knowledge_id,"Trigger":x.candidate.record.trigger_event,"Similarity":f"{x.candidate.similarity:.1%}","Verdict":x.verdict.replace('_',' ').title(),"Reasoning":x.explanation} for x in review.candidates]

def hlib_pdf_path(record):
    """Resolve an HLIB source filename without allowing paths outside the PDF folder."""
    if record.source_name != "HLIB Research" or not record.source_file:
        return None
    candidate = (HLIB_SOURCE_DIR / Path(record.source_file).name).resolve()
    if candidate.parent != HLIB_SOURCE_DIR or candidate.suffix.lower() != ".pdf" or not candidate.is_file():
        return None
    return candidate

@st.cache_data(show_spinner=False)
def pdf_bytes(path):
    return Path(path).read_bytes()

def source_control(record, key_suffix):
    pdf_path = hlib_pdf_path(record)
    if pdf_path:
        key = hashlib.sha1(f"{pdf_path}:{key_suffix}".encode()).hexdigest()
        st.download_button(
            "Download original HLIB PDF",
            data=pdf_bytes(str(pdf_path)),
            file_name=pdf_path.name,
            mime="application/pdf",
            key=f"hlib_pdf_{key}",
        )
    elif record.source_name == "HLIB Research":
        st.caption(f"Original PDF is unavailable locally: {record.source_file or 'filename not recorded'}")
    elif record.source_link:
        st.link_button("Open original source",record.source_link)

@st.dialog("Knowledge record evidence", width="large")
def record_dialog(x):
    r=x.candidate.record; st.subheader(r.knowledge_id); st.caption(f"{r.source_name} · {r.source_title} · {r.source_date}"); st.write(x.explanation)
    for e in r.evidence:
        st.markdown(f'<div class="quote">“{escape(e.quote)}”</div>',unsafe_allow_html=True)
        if e.translation: st.write("English:",e.translation)
        st.caption(e.location)
    source_control(r,f"dialog:{r.knowledge_id}")
    st.markdown("#### Raw metadata"); st.json(r.model_dump(mode="json"),expanded=False)

def relevance_view(review,key):
    st.dataframe(relevance_rows(review),hide_index=True,width="stretch",height=380)
    byid={x.candidate.record.knowledge_id:x for x in review.candidates}
    if not byid:return
    selected=st.selectbox("Inspect a record",list(byid),format_func=lambda v:f"{v} · {byid[v].verdict.replace('_',' ').title()}",key=key+"select")
    if st.button("Open evidence and raw metadata",key=key+"open"):record_dialog(byid[selected])

def impact_view(x,label):
    level="Source-grounded inference" if x.support_level=="source_grounded_inference" else "Direct source support"; cls="inference" if x.support_level=="source_grounded_inference" else "direct"
    ids=''.join(f'<span class="pill">{escape(i)}</span>' for i in x.supporting_knowledge_ids); steps=''.join(f'<li>{escape(s)}</li>' for s in x.causal_steps)
    companies=''.join(f'<span class="pill">Company: {escape(c.company_name)}{f" ({escape(c.ticker)})" if c.ticker else ""}</span>' for c in getattr(x,"impacted_companies",[]))
    st.markdown(f'<div class="impact"><b>{escape(label)} · <span class="{x.impact_direction}">{x.impact_direction.title()}</span></b><br><span class="badge {cls}">{level}</span><p>{escape(x.conclusion)}</p><ol>{steps}</ol>{companies}{ids}</div>',unsafe_allow_html=True)

def source_view(a):
    st.markdown(f'<div class="panel"><div class="source">{a.source_name}</div><div class="kind">{"Professional analyst commentary" if a.source_name=="Chen" else "Institutional research"}</div><p>{escape(a.source_summary)}</p>',unsafe_allow_html=True)
    if a.no_supported_conclusion:st.markdown('<div class="empty">No direct evidence or applicable rule supports a directional conclusion.</div>',unsafe_allow_html=True)
    for x in a.sector_impacts:impact_view(x,f"{x.impacted_industry} · {x.impacted_sector}" if x.impacted_industry else x.impacted_sector)
    for x in a.company_impacts:impact_view(x,x.company_name+(f" ({x.ticker})" if x.ticker else ""))
    if a.market_context:
        st.markdown("**Relevant market context**")
        for x in a.market_context:st.markdown(f'<div class="context">• {escape(x.summary)}</div>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

def evidence_view(bundle):
    with st.expander("Evidence used in final verdicts"):
        records=bundle["records"]
        for source,a in bundle["synthesis"].analyses.items():
            st.markdown(f"#### {source}")
            for impact_index,x in enumerate(list(a.sector_impacts)+list(a.company_impacts)):
                shown=False
                for e in x.transcript_evidence:
                    tr=bundle["transcripts"].get(e.video_id)
                    if tr:
                        st.markdown(f'<div class="quote">“{escape(tr.exact_text(e.start_seconds,e.end_seconds))}”</div>',unsafe_allow_html=True);st.write("English:",e.translation);st.caption(f"{tr.title} · {format_seconds(e.start_seconds)}–{format_seconds(e.end_seconds)}");shown=True
                        if tr.url:st.link_button("Open video at timestamp",timestamped_video_url(tr.url,e.start_seconds))
                if not shown:
                    for kid in x.supporting_knowledge_ids:
                        r=records.get(kid)
                        if r:
                            for e in r.evidence:st.markdown(f'<div class="quote">“{escape(e.quote)}”</div>',unsafe_allow_html=True);st.caption(f"{r.source_title} · {e.location}")
                            source_control(r,f"evidence:{source}:{impact_index}:{kid}")

def comparison_view(b):
    c=b["comparison"].comparison;st.markdown(f'<div class="compare"><b>Overall comparison</b><p>{escape(c.overall_summary)}</p></div>',unsafe_allow_html=True)
    l,r=st.columns(2)
    with l:
        st.markdown("#### Agreements")
        if c.agreements:
            for item in c.agreements: st.success(item)
        else: st.caption("None identified.")
    with r:
        st.markdown("#### Differences")
        if c.differences:
            for item in c.differences: st.info(item)
        else: st.caption("None identified.")
    st.write("**Evidence comparison:**",c.evidence_comparison);st.markdown("#### Interactive causal map");render_interactive_graph(b["graph"])

def completed(b):
    anchor(1)
    with st.expander("✓ Stage 1 · Analyse the user query"):event_view(b["event_analysis"])
    anchor(2)
    with st.expander("✓ Stage 2 · Retrieve knowledge records"):st.success(f"Retrieved {len(b['chen'])} Chen and {len(b['hlib'])} HLIB records.")
    anchor(3)
    with st.expander("✓ Stage 3 · Determine relevancy of retrieved records"):relevance_view(b["review"],"cached")
    anchor(4)
    with st.expander("✓ Stage 4 · Retrieve further context from high-similarity sources"):
        st.write(f"Full transcripts collected from {len(b['transcripts'])} Direct Chen source(s).")
        for transcript in b["transcripts"].values(): st.caption(transcript.title)
    anchor(5)
    with st.expander("✓ Stage 5 · Generate final verdicts",expanded=True):
        l,r=st.columns(2)
        with l:source_view(b["synthesis"].analyses["Chen"])
        with r:source_view(b["synthesis"].analyses["HLIB Research"])
        if b["synthesis"].used_fallback:
            st.warning(
                b["synthesis"].status_message
                or "Final LLM synthesis was unavailable; a deterministic fallback was shown."
            )
        evidence_view(b)
    anchor(6)
    with st.expander("✓ Stage 6 · Compare Chen and HLIB results",expanded=True):comparison_view(b)

quick_launch_url = str(st.query_params.get("article_url", "")).strip()
quick_launch_pending = bool(
    quick_launch_url
    and st.session_state.get("completed_quick_launch_url") != quick_launch_url
)

with st.container(key="top_header"):
    left,right=st.columns([.75,1.25])
    with left:st.markdown('<div class="brand"><div><div class="title">Stock Investment Event-Impact Agent</div><div class="sub">An LLM-based agent for analysing how news events may affect Malaysian industry sectors and listed companies, using Alfred Chen commentary and HLIB Research as separate knowledge bases.</div></div></div>',unsafe_allow_html=True)
    with right:
        mode=st.radio(
            "Event source", ["Type event","Paste article URL","Upload PDF"],
            index=1 if quick_launch_url else 0,
            horizontal=True, label_visibility="collapsed"
        );text=url="";upload=None
        if mode=="Type event":text=st.text_input("Event",placeholder="Describe an event in English",label_visibility="collapsed")
        elif mode=="Paste article URL":url=st.text_input(
            "URL", value=quick_launch_url,
            placeholder="Paste a readable article URL", label_visibility="collapsed"
        )
        else:upload=st.file_uploader("PDF",type=["pdf"],label_visibility="collapsed")
        manual_click=st.button("Analyse event",type="primary",disabled=not bool(text.strip() or url.strip() or upload),width="stretch")
    navbox=st.empty()

clicked = manual_click or quick_launch_pending

if not clicked and "analysis_bundle" not in st.session_state:
    st.subheader("How your event will be analysed");flow();st.stop()
nav(navbox)
if not clicked:
    completed(st.session_state.analysis_bundle);st.stop()

anchor(1)
with st.status("Stage 1 · Analysing the user query…",expanded=True) as s:
    src=from_typed_text(text) if mode=="Type event" else from_url(url) if mode=="Paste article URL" else from_pdf(upload.getvalue(),upload.name);analysis=analyse_event(src.text);event_view(analysis);s.update(label="✓ Stage 1 · User query analysed",state="complete",expanded=False)
anchor(2)
with st.status("Stage 2 · Retrieving knowledge records…",expanded=True) as s:
    chen=retrieve_diverse_candidates(analysis.retrieval_queries,"Chen",per_query_limit=4,total_limit=10);hlib=retrieve_diverse_candidates(analysis.retrieval_queries,"HLIB Research",per_query_limit=4,total_limit=10);st.success(f"Retrieved {len(chen)} Chen and {len(hlib)} HLIB records.");s.update(label="✓ Stage 2 · Knowledge retrieved",state="complete",expanded=False)
anchor(3)
with st.status("Stage 3 · Determining relevancy…",expanded=True) as s:
    review=review_relevance(src.text,analysis.event_summary,analysis.retrieval_queries,chen+hlib);relevance_view(review,"live");s.update(label="✓ Stage 3 · Relevancy determined",state="complete",expanded=False)
direct=[x.candidate.record for x in review.candidates if x.direct];applicable=[x.candidate.record for x in review.candidates if x.applicable_rule];background=[x.candidate.record for x in review.candidates if x.verdict=="general_background"];records={r.knowledge_id:r for r in direct+applicable+background}
anchor(4)
with st.status("Stage 4 · Retrieving further source context…",expanded=True) as s:
    transcripts,warnings=load_direct_chen_transcripts(direct);st.write(f"Full transcripts collected from {len(transcripts)} Direct Chen source(s).")
    for transcript in transcripts.values(): st.caption(transcript.title)
    s.update(label="✓ Stage 4 · Source context retrieved",state="complete",expanded=False)
anchor(5)
with st.status("Stage 5 · Generating final verdicts…",expanded=True) as s:
    synthesis=synthesize_impacts(src.text,analysis.event_summary,direct,applicable,background,transcripts);l,r=st.columns(2)
    with l:source_view(synthesis.analyses["Chen"])
    with r:source_view(synthesis.analyses["HLIB Research"])
    if synthesis.used_fallback:
        st.warning(synthesis.status_message or "Final LLM synthesis was unavailable; a deterministic fallback was shown.")
        s.update(label="⚠ Stage 5 · Fallback verdicts shown",state="complete",expanded=True)
    else:
        s.update(label="✓ Stage 5 · Verdicts generated",state="complete",expanded=False)
anchor(6)
with st.status("Stage 6 · Comparing agents and constructing graph…",expanded=True) as s:
    comparison=compare_source_results(analysis.event_summary,synthesis.analyses);graph=build_graph_data(analysis.event_summary,synthesis.analyses);bundle={"event_analysis":analysis,"chen":chen,"hlib":hlib,"review":review,"transcripts":transcripts,"warnings":warnings,"synthesis":synthesis,"comparison":comparison,"graph":graph,"records":records};comparison_view(bundle);s.update(label="✓ Stage 6 · Comparison ready",state="complete",expanded=True)
st.session_state.analysis_bundle=bundle
if quick_launch_pending:
    st.session_state.completed_quick_launch_url = quick_launch_url
st.markdown('<div class="disclaimer">Research prototype · Not investment advice</div>',unsafe_allow_html=True)
