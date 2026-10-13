from pathlib import Path
from core.ingestion import list_docs, load_doc, _default_data_dirs
from core.vector_store import VectorStore
def ingest_documents():
    docs_dir = _default_data_dirs()
    pdf_files = list_docs(docs_dir)
    if not pdf_files:
        print("No documents found in the docs/ directory. Exiting.")
        return

    vs = VectorStore()
    if not vs.initialize():
        print("❌ Failed to initialize Vector Store.")
        return

    doc_ids = {filename: f"d_{Path(filename).stem}" for filename in pdf_files}
    try:
        existing_ids = vs.existing_doc_ids(list(doc_ids.values()))
    except Exception as e:
        print(f"Could not check existing Pinecone documents; stopping ingestion: {e}")
        return

    new_files = [filename for filename, doc_id in doc_ids.items() if doc_id not in existing_ids]
    if not new_files:
        print("All PDFs are already present in Pinecone.")
        return

    print(f"1. Loading and parsing {len(new_files)} new documents...")
    docs = load_doc(docs_dir, only_files=new_files)
    if not docs:
        print("No new documents could be loaded. Exiting.")
        return

    print(f"Loaded {len(docs)} documents. Pushing to Pinecone...")
    if vs.add_docs(docs):
        print("✅ Ingestion complete!")
    else:
        print("❌ Failed to push documents to Pinecone.")
def search(query: str):
    print(f"\nSearching for: '{query}'")
    vs = VectorStore()
    if vs.initialize():
        results = vs.search_docs(query, top_k=3)
        if not results:
            print("No results found.")
            return
            
        for i, r in enumerate(results, 1):
            doc_id = r.get('metadata', {}).get('doc_id', 'Unknown')
            name = r.get('metadata', {}).get('name', 'Unknown')
            score = r.get('score', 0)
            print(f"\nResult {i} (Score: {score:.4f})")
            print(f"Document ID: {doc_id} | Name: {name}")
            print(f"Content Preview: {r.get('page_content', '')[:200]}...")
if __name__ == "__main__":
# 1. First, run this to ingest your PDFs (Comment it out after you run it once)
    ingest_documents()
# 2. Then, you can run searches like this:
    search("How do I enroll iPhone to intune?")