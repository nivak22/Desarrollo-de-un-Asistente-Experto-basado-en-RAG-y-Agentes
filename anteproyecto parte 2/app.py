from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

st.set_page_config(page_title="RAG Técnico Asistente", page_icon="🤖", layout="centered")
st.title("🤖 Asistente Experto (RAG)")
st.markdown(
    "Consulta manuales técnicos. Las respuestas están ancladas estrictamente a los documentos base."
)

api_key = st.text_input(
    "Clave de Google Gemini",
    value=os.getenv("GOOGLE_API_KEY", ""),
    type="password",
    help="La clave se usa durante esta sesión para consultar Gemini. No la compartas ni la incluyas en capturas.",
)
if not api_key:
    st.info("Ingresa tu propia clave de Google Gemini para comenzar.")
    st.stop()


@st.cache_resource
def load_vector_store():
    if not os.path.exists(str(ROOT_DIR / "chroma_db")):
        from src.build_index import build_vector_store

        vector_store = build_vector_store()
        if vector_store is None:
            raise RuntimeError("No se pudo crear el índice. Agrega documentos a la carpeta data.")
        return vector_store

    from src.rag_core import get_vector_store

    return get_vector_store()


from src.rag_core import get_rag_chain

rag_chain = get_rag_chain(api_key=api_key, vector_store=load_vector_store())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

user_input = st.chat_input("Escribe tu pregunta aquí...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en la base de conocimientos..."):
            response = rag_chain.invoke(
                {
                    "input": user_input,
                    "chat_history": st.session_state.chat_history,
                }
            )

            answer = response["answer"]
            source_docs = response.get("context", [])
            st.write(answer)

            if source_docs and "La información no se encuentra" not in answer:
                with st.expander("Ver fuentes documentales"):
                    for idx, doc in enumerate(source_docs, start=1):
                        source = doc.metadata.get("source", "Desconocido")
                        page = doc.metadata.get("page", "N/A")
                        st.markdown(f"**Fuente {idx}:** `{source}` (Página {page})")
                        st.caption(f'"{doc.page_content[:200]}..."')

    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=answer))
