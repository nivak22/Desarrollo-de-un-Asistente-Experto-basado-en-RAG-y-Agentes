from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
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

    raise ValueError("El modelo devolvió una respuesta sin contenido de texto.")


class SimpleRAGChain:
    def __init__(self, vector_store, llm):
        self.vector_store = vector_store
        self.llm = llm

    def invoke(self, inputs):
        question = str(inputs.get("input", "")).strip()
        if not question:
            return {"answer": "No hay pregunta para responder.", "context": []}

        context_docs = self.vector_store.similarity_search(question, k=4)
        context_text = "\n\n".join(doc.page_content for doc in context_docs)

        system_prompt = (
            "Eres un asistente técnico experto. Responde utilizando ÚNICAMENTE los fragmentos "
            "de contexto proporcionados. Si la respuesta no aparece en el contexto, responde exactamente: "
            "'La información no se encuentra en la base de conocimientos proporcionada.' "
            "No inventes ni adivines datos.\n\nContexto:\n"
            f"{context_text}"
        )

        response = self.llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=question),
            ]
        )
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


def get_rag_chain(api_key: str | None = None, vector_store=None):
    """Crea una cadena RAG usando la clave proporcionada o GOOGLE_API_KEY."""
    load_dotenv(ROOT_DIR / ".env")

    api_key = api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Falta la clave de Google Gemini.")

    if vector_store is None:
        vector_store = get_vector_store()

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0.1,
        google_api_key=api_key,
    )
    return SimpleRAGChain(vector_store, llm)
