import streamlit as st

from compli_pipeline import CompliBotPipeline


st.set_page_config(page_title="CompliBot MVP", page_icon="📋", layout="wide")

st.title("📋 CompliBot MVP")
st.markdown(
    "Upload SOP, quality, and compliance PDFs. Ask grounded questions and get structured compliance answers."
)

if "pipeline" not in st.session_state:
    st.session_state.pipeline = CompliBotPipeline()

if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False

if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = []

if "last_ingest_count" not in st.session_state:
    st.session_state.last_ingest_count = 0

pipeline = st.session_state.pipeline

with st.sidebar:
    st.header("Document Upload")

    uploaded_files = st.file_uploader(
        "Upload compliance PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )

    if st.button("Reset Session"):
        pipeline.reset_collection()
        st.session_state.docs_loaded = False
        st.session_state.processed_docs = []
        st.session_state.last_ingest_count = 0
        st.success("Session reset complete.")

    if uploaded_files:
        st.subheader("Files Selected for Next Ingest")
        for f in uploaded_files:
            st.write(f"- {f.name}")

    if st.button("Ingest Documents"):
        if uploaded_files:
            with st.spinner("Processing and indexing documents..."):
                total_chunks, processed_docs = pipeline.ingest_documents(uploaded_files)

            st.session_state.docs_loaded = len(processed_docs) > 0
            st.session_state.processed_docs = processed_docs
            st.session_state.last_ingest_count = pipeline.count_indexed_chunks()

            st.success(
                f"Ingested {len(processed_docs)} document(s). "
                f"Collection now has {st.session_state.last_ingest_count} chunks."
            )
        else:
            st.warning("Please upload at least one PDF before ingesting.")

    st.subheader("Currently Ingested Documents")
    if st.session_state.processed_docs:
        for doc in st.session_state.processed_docs:
            st.write(f"- {doc}")
        st.caption(f"Indexed chunks in collection: {st.session_state.last_ingest_count}")
    else:
        st.caption("No documents currently ingested.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Ask a Compliance Question")
    user_question = st.text_area(
        "Enter your question",
        placeholder="Example: What does this SOP say about deviation handling?",
        height=120,
    )

    ask_clicked = st.button("Get Compliance Answer")

    if ask_clicked:
        if not st.session_state.docs_loaded:
            st.warning("Please upload and ingest documents first.")
        elif not user_question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving structured answer from compliance documents..."):
                retrieved = pipeline.retrieve_relevant_chunks(user_question, top_k=4)
                result = pipeline.synthesize_answer(user_question, retrieved)

            st.markdown("## Answer Summary")
            st.write(result["answer_summary"])

            st.markdown("## Procedure / Guidance")
            st.write(result["procedure_guidance"])

            st.markdown("## Key Requirements")
            if result["key_requirements"]:
                for item in result["key_requirements"]:
                    st.write(f"- {item}")
            else:
                st.write("No explicit key requirements identified.")

            st.markdown("## Primary Source")
            st.info(result["source"])

            st.markdown("## Supporting Evidence")
            if result["evidence"]:
                for idx, item in enumerate(result["evidence"], start=1):
                    label = f"Evidence {idx} — {item['source']} (chunk {item['chunk_index']})"
                    if "distance" in item:
                        label += f" | distance={item['distance']:.4f}"
                    with st.expander(label):
                        st.write(item["text"])
            else:
                st.write("No evidence available.")

            st.markdown("## Compliance Interpretation Note")
            st.warning(result["compliance_note"])

            st.markdown("## Debug Info")
            st.write(f"Retrieved chunk count: {len(retrieved)}")
            if retrieved:
                st.write("Top retrieved sources:")
                for r in retrieved:
                    st.write(
                        f"- {r['source']} | chunk {r['chunk_index']} "
                        f"| group={r.get('doc_group', 'n/a')} | distance={r['distance']:.4f}"
                    )

with col2:
    st.subheader("Question Context")
    st.markdown(
        """
        **Supported question styles**
        - Definitions
        - Procedures
        - Policy requirements
        - Training guidance
        - Escalation / deviation / CAPA questions
        """
    )

    if user_question.strip():
        qtype = pipeline.classify_question(user_question)
        st.markdown("### Detected Question Type")
        st.success(qtype)