from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DB_DIR = ROOT_DIR / "chroma_db"


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """Carga documentos PDF y texto desde el directorio de datos."""
    documents: list[Document] = []

    if not data_dir.exists():
        print(f"No se encontró el directorio de datos: {data_dir}")
        return documents

    for file_path in sorted(data_dir.iterdir()):
        if file_path.is_dir():
            continue

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
            documents.extend(loader.load())
        elif suffix in {".txt", ".md"}:
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents.extend(loader.load())

    return documents


def build_vector_store(
    data_dir: Path = DATA_DIR,
    persist_directory: Path = DB_DIR,
):
    """Construye la base de datos vectorial a partir del corpus disponible."""
    load_dotenv(ROOT_DIR / ".env")

    docs = load_documents(data_dir)
    if not docs:
        print(f"No se encontraron documentos en {data_dir}. Agrega archivos PDF o TXT.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Se generaron {len(chunks)} fragmentos.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_directory),
    )
    print(f"Índice vectorial creado y guardado en {persist_directory}")
    return vector_store


if __name__ == "__main__":
    build_vector_store()
