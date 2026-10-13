from pathlib import Path

import streamlit as st

from core.ingestion import _default_data_dirs, list_docs, load_doc
from core.vector_store import VectorStore
from utils.config import PINECONE_API_KEY


st.set_page_config(page_title="Document Search", layout="centered")
st.title("Document Search")
st.caption("Search your PDF knowledge base with semantic retrieval.")

docs_dir = _default_data_dirs()
pdf_files = list_docs(docs_dir)


@st.cache_resource(show_spinner=False)
def get_vector_store() -> VectorStore | None:
    store = VectorStore()
    return store if store.initialize() else None


def show_vector_store_error() -> None:
    if not PINECONE_API_KEY:
        st.error("Set PINECONE_API_KEY in your .env file to connect to the search index.")
    else:
        st.error("Could not connect to Pinecone. Check your configuration and try again.")


with st.sidebar:
    st.subheader("PDF library")
    st.caption(f"Source folder: {Path(docs_dir).name}/")
    st.write(f"{len(pdf_files)} PDF{'s' if len(pdf_files) != 1 else ''} found")

    with st.expander("View PDFs", expanded=False):
        if pdf_files:
            for filename in pdf_files:
                st.write(filename)
        else:
            st.write("Add PDF files to the docs folder to get started.")

    ingest_clicked = st.button(
        "Index new PDFs",
        type="secondary",
        disabled=not pdf_files,
        use_container_width=True,
    )

if ingest_clicked:
    if not PINECONE_API_KEY:
        show_vector_store_error()
    else:
        with st.spinner("Connecting to the document index..."):
            vector_store = get_vector_store()

        if vector_store is None:
            show_vector_store_error()
        else:
            doc_ids = {
                filename: f"d_{Path(filename).stem}" for filename in pdf_files
            }
            try:
                existing_ids = vector_store.existing_doc_ids(list(doc_ids.values()))
            except Exception:
                st.error("Could not check existing documents in Pinecone.")
            else:
                new_files = [
                    filename
                    for filename, doc_id in doc_ids.items()
                    if doc_id not in existing_ids
                ]

                if not new_files:
                    st.info("All PDFs are already indexed.")
                else:
                    progress = st.progress(0, text="Preparing documents...")

                    def update_progress(current: int, total: int, filename: str) -> None:
                        progress.progress(
                            current / total,
                            text=f"Loading {filename} ({current}/{total})",
                        )

                    documents = load_doc(
                        docs_dir,
                        only_files=new_files,
                        progress=update_progress,
                    )
                    if not documents:
                        st.warning("No readable documents were found.")
                    elif vector_store.add_docs(documents):
                        progress.progress(1.0, text="Indexing complete")
                        st.success(f"Indexed {len(documents)} PDF(s).")
                    else:
                        st.error("Could not add the documents to Pinecone.")

st.subheader("Search")
with st.form("search_form"):
    query = st.text_input("Question", placeholder="How do I enroll an iPhone?")
    top_k = st.slider("Results", min_value=1, max_value=10, value=3)
    search_clicked = st.form_submit_button("Search documents", type="primary")

if search_clicked:
    if not query.strip():
        st.warning("Enter a question to search the documents.")
    elif not PINECONE_API_KEY:
        show_vector_store_error()
    else:
        with st.spinner("Searching your documents..."):
            vector_store = get_vector_store()
            results = (
                vector_store.search_docs(query.strip(), top_k=top_k)
                if vector_store is not None
                else []
            )

        if vector_store is None:
            show_vector_store_error()
        elif not results:
            st.info("No matching documents found.")
        else:
            st.caption(f"{len(results)} matching result(s)")
            for index, result in enumerate(results, start=1):
                metadata = result.get("metadata") or {}
                name = metadata.get("name") or metadata.get("filename") or "Document"
                source = metadata.get("filename") or metadata.get("source") or "Unknown file"
                score = result.get("score")

                st.markdown(f"**{index}. {name}**")
                score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "N/A"
                st.caption(f"{source} · Relevance: {score_text}")
                with st.expander("View matching text"):
                    st.text(result.get("page_content") or "No text preview available.")