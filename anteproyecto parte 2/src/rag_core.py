from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_DIR = ROOT_DIR / "chroma_db"


def _extract_response_text(content: object) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_blocks = [
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ]
        if text_blocks:
            return "".join(text_blocks)

    if hasattr(content, "text") and isinstance(content.text, str):
        return content.text

    raise ValueError("El modelo devolvió una respuesta sin contenido textual.")


class SimpleRAGChain:
    def __init__(self, vector_store, llm, top_k: int = 4):
        self.vector_store = vector_store
        self.llm = llm
        self.top_k = top_k

    def invoke(self, inputs):
        question = str(inputs.get("input", "")).strip()
        if not question:
            return {"answer": "No hay pregunta para responder.", "context": []}

        chat_history = list(inputs.get("chat_history", []) or [])
        context_docs = self.vector_store.similarity_search(question, k=self.top_k)
        context_text = "\n\n".join(doc.page_content for doc in context_docs)

        system_prompt = (
            "Eres un asistente técnico experto en manuales y mantenimiento. "
            "Responde exclusivamente con la información presente en el contexto proporcionado. "
            "Si la respuesta no aparece textualmente en el contexto, responde exactamente: "
            "'La información no se encuentra en la base de conocimientos proporcionada.' "
            "No inventes, no asumas ni improvises datos. "
            "Cuando la evidencia exista, menciona la fuente y la página cuando la conozcas.\n\n"
            f"Contexto:\n{context_text}"
        )

        messages = [SystemMessage(content=system_prompt)]
        messages.extend(chat_history)
        messages.append(HumanMessage(content=question))

        response = self.llm.invoke(messages)
        answer = _extract_response_text(getattr(response, "content", None))
        return {"answer": answer, "context": context_docs}


def get_vector_store():
    """Carga la base vectorial local con el modelo de embeddings del proyecto."""
    load_dotenv(ROOT_DIR / ".env")

    if not DB_DIR.exists():
        raise FileNotFoundError(
            "No existe la base de conocimiento. Ejecuta 'python src/build_index.py' antes de iniciar el asistente."
        )

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)


def get_rag_chain(api_key: str | None = None, vector_store=None, top_k: int = 4):
    """Crea una cadena RAG usando la clave proporcionada o GOOGLE_API_KEY."""
    load_dotenv(ROOT_DIR / ".env")

    api_key = api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Falta la clave de Google Gemini.")

    if vector_store is None:
        vector_store = get_vector_store()

    model_name = os.getenv("GOOGLE_MODEL", "gemini-3.8-flash")
    if model_name.startswith("models/"):
        model_name = model_name.replace("models/", "")

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.1,
        google_api_key=api_key,
    )
    return SimpleRAGChain(vector_store, llm, top_k=top_k)
