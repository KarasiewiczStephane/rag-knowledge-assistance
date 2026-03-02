"""Streamlit chat interface for the RAG Knowledge Assistant."""

import logging
import tempfile
from pathlib import Path

import streamlit as st

from src.rag_pipeline import RAGPipeline
from src.utils.config import load_config
from src.utils.logger import configure_logging

logger = logging.getLogger(__name__)


def _init_pipeline() -> RAGPipeline:
    """Initialize or retrieve the RAG pipeline from session state.

    Returns:
        Configured RAGPipeline instance.
    """
    if "pipeline" not in st.session_state:
        config = load_config()
        configure_logging(config)
        st.session_state.pipeline = RAGPipeline(config)
    return st.session_state.pipeline


def _init_session(pipeline: RAGPipeline) -> str:
    """Initialize or retrieve the conversation session.

    Args:
        pipeline: The RAG pipeline instance.

    Returns:
        Active session ID.
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = pipeline.start_new_session()
    return st.session_state.session_id


def _render_sidebar(pipeline: RAGPipeline) -> None:
    """Render the document management sidebar.

    Args:
        pipeline: The RAG pipeline instance.
    """
    st.sidebar.title("Document Management")

    uploaded_files = st.sidebar.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "md", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=f".{uploaded_file.name.split('.')[-1]}",
            ) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            with st.sidebar.status(f"Ingesting {uploaded_file.name}..."):
                try:
                    count = pipeline.ingest_document(tmp_path)
                    if count > 0:
                        st.sidebar.success(
                            f"Ingested {uploaded_file.name}: {count} chunks"
                        )
                    else:
                        st.sidebar.warning(f"{uploaded_file.name}: duplicate or empty")
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

    st.sidebar.divider()
    st.sidebar.subheader("Indexed Documents")

    docs = pipeline.vector_store.list_documents()
    if docs:
        for doc in docs:
            filename = doc["source_file"].split("/")[-1]
            col1, col2 = st.sidebar.columns([3, 1])
            col1.write(f"**{filename}** ({doc['chunk_count']} chunks)")
            if col2.button("Del", key=f"del_{doc['source_file']}"):
                pipeline.vector_store.delete_document(doc["source_file"])
                st.rerun()
    else:
        st.sidebar.info("No documents indexed yet.")

    st.sidebar.divider()
    st.sidebar.subheader("Session")

    if st.sidebar.button("New Conversation"):
        st.session_state.session_id = pipeline.start_new_session()
        st.session_state.messages = []
        st.rerun()

    if st.sidebar.button("Export Conversation"):
        if "session_id" in st.session_state:
            path = pipeline.memory.save_session(st.session_state.session_id)
            st.sidebar.success(f"Saved to {path}")


def _render_chat(pipeline: RAGPipeline, session_id: str) -> None:
    """Render the main chat interface.

    Args:
        pipeline: The RAG pipeline instance.
        session_id: Active conversation session ID.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("View Sources"):
                    st.markdown(msg["sources"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = pipeline.process_query(prompt, session_id=session_id)
                    st.markdown(response.answer)

                    if response.low_confidence:
                        st.warning(
                            "Low confidence answer. "
                            "The retrieved context may not "
                            "fully address your question."
                        )

                    with st.expander("View Sources"):
                        st.markdown(response.sources_markdown)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response.answer,
                            "sources": response.sources_markdown,
                        }
                    )
                except Exception as e:
                    error_msg = f"Error processing query: {e}"
                    st.error(error_msg)
                    logger.error(error_msg)


def main() -> None:
    """Run the Streamlit dashboard application."""
    st.set_page_config(
        page_title="RAG Knowledge Assistant",
        page_icon="📚",
        layout="wide",
    )

    st.title("RAG Knowledge Assistant")
    st.caption("Upload documents and ask questions with source citations")

    pipeline = _init_pipeline()
    session_id = _init_session(pipeline)

    _render_sidebar(pipeline)
    _render_chat(pipeline, session_id)


if __name__ == "__main__":
    main()
