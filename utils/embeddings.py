import asyncio
import logging
from contextlib import contextmanager
from typing import Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.utils import is_quota_error

logger = logging.getLogger(__name__)

def _ensure_event_loop() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@contextmanager
def _quiet_progress_bars():
    restore = []
    try:
        try:
            from huggingface_hub.utils import (
                disable_progress_bars, enable_progress_bars,
            )
            disable_progress_bars()
            restore.append(enable_progress_bars)
        except Exception:
            pass
        try:
            from transformers.utils import logging as hf_logging
            hf_logging.disable_progress_bar()
            restore.append(hf_logging.enable_progress_bar)
        except Exception:
            pass
        yield
    finally:
        for restore_fn in restore:
            try:
                restore_fn()
            except Exception:
                pass

def get_embeddings() -> Optional[HuggingFaceEmbeddings]:
    try:
        _ensure_event_loop()

        # return GoogleGenerativeAIEmbeddings(
        #     model="models/embedding-001",  # Google's text embedding model
        #     google_api_key=GOOGLE_API_KEY,
        # )

        try:
            # Cached load: nothing to report, so suppress the weight-loading
            # progress bar. It is left enabled on the download path below,
            # where a first-time user genuinely wants to see progress.
            with _quiet_progress_bars():
                return HuggingFaceEmbeddings(
                    model_name=MODEL_NAME,
                    model_kwargs={"local_files_only": True},
                )
        except Exception:
            logger.info(
                f"{MODEL_NAME} not in the local cache — downloading it once."
            )
            return HuggingFaceEmbeddings(model_name=MODEL_NAME)

    except Exception as e:
        if is_quota_error(e):
            logger.warning("API quota exceeded. Vector search disabled.")
        else:
            logger.exception("Embeddings initialization failed")
        return None