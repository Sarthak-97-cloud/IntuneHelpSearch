import os
import time
from pathlib import Path
from typing import Callable, Iterable
from langchain.schema import Document
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.utils import get_logger, load_pdf

logger = get_logger(__name__)

def _get_parser():
    try:
        from core.parsing import DocParser
        return DocParser()
    except Exception as e:
        logger.warning(f"DocParser unavailable, falling back to basic metadata: {e}")
        return None

def _try_parse_doc(text: str, doc_id: str, parser=None) -> dict:
    if parser is None:
        parser = _get_parser()
    if parser is None:
        return {}
    try:
        return parser.parse_doc(text, doc_id)
    except Exception as e:
        logger.warning(f"Doc parsing failed for document {doc_id}: {e}")
        return {}

def build_doc(file_path: str, filename: str = None, parser=None) -> Document | None:
    filename = filename or Path(file_path).name
    text = load_pdf(file_path)
    if not text:
        return None

    doc_id = f"d_{Path(filename).stem}"
    fallback_name = Path(filename).stem.replace('_', ' ').title()

    parsed = _try_parse_doc(text, doc_id, parser=parser)

    return Document(
        page_content=text,
        metadata={
            "source": file_path,
            "filename": filename,
            "doc_id": doc_id,
            "name": parsed.get('name') or fallback_name
        }
    )

def list_docs(directory: str) -> list[str]:
    if not directory or not os.path.isdir(directory):
        logger.warning(f"Documents directory not found: {directory}")
        return []
    return sorted(f for f in os.listdir(directory) if f.lower().endswith('.pdf'))

def load_doc(directory: str, only_files: Iterable[str] = None,
                 progress: Callable[[int, int, str], None] = None) -> list[Document]:

    available = list_docs(directory)
    if not available:
        return []

    targets = [f for f in available if f in set(only_files)] if only_files is not None else available
    if not targets:
        return []

    # One parser (and one LLM client) for the whole run.
    parser = _get_parser()

    docs = []
    total = len(targets)
    for i, file in enumerate(targets, start=1):
        file_path = os.path.join(directory, file)
        try:
            if progress:
                progress(i, total, file)
            doc = build_doc(file_path, filename=file, parser=parser)
            if doc is not None:
                docs.append(doc)
                time.sleep(4.5)
        except Exception as e:
            logger.error(f"Failed to load doc {file_path}: {e}")
            continue
    return docs

def _default_data_dirs():
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    docs_dir = project_root / "docs"
    return str(docs_dir)

def _print_sample_doc(docs, label: str, max_display: int = 3):
    print(f"\n{label}: {len(docs)} found")
    for i, doc in enumerate(docs[:max_display]):
        meta = getattr(doc, 'metadata', {}) or {}
        print(f"[{i+1}] id: {meta.get('doc_id', 'n/a')} | filename: {meta.get('filename', meta.get('title', 'n/a'))}")
        snippet = (getattr(doc, 'page_content', '') or '')[:200]
        flat_snippet = snippet.replace('\n', ' ')[:180]
        print(f"    snippet: {flat_snippet}...")

if __name__ == '__main__':
    data_dirs = _default_data_dirs()
    docs = load_doc(data_dirs)
    _print_sample_doc(docs, "Loaded documents")

