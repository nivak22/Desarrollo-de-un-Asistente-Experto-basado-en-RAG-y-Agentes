from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from docx import Document as DocxDocument

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DB_DIR = ROOT_DIR / "chroma_db"


class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str):
        text = " ".join(data.split())
        if text:
            self.parts.append(text)

    def get_text(self) -> str:
        return "\n".join(self.parts)


def _attach_source_metadata(doc: Document, file_path: Path) -> Document:
    doc.metadata.setdefault("source", file_path.name)
    doc.metadata.setdefault("file_type", file_path.suffix.lower().lstrip("."))
    doc.metadata.setdefault("document_name", file_path.name)
    return doc


def _read_docx(file_path: Path) -> str:
    doc = DocxDocument(str(file_path))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(paragraphs)


def _read_html(file_path: Path) -> str:
    parser = HTMLTextExtractor()
    parser.feed(file_path.read_text(encoding="utf-8", errors="ignore"))
    return parser.get_text()


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """Carga documentos PDF, DOCX, HTML, TXT y Markdown desde el corpus de entrada."""
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
            loaded_docs = loader.load()
            for doc in loaded_docs:
                documents.append(_attach_source_metadata(doc, file_path))
        elif suffix in {".txt", ".md"}:
            loader = TextLoader(str(file_path), encoding="utf-8")
            loaded_docs = loader.load()
            for doc in loaded_docs:
                documents.append(_attach_source_metadata(doc, file_path))
        elif suffix == ".docx":
            text = _read_docx(file_path)
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_path.name,
                            "file_type": "docx",
                            "document_name": file_path.name,
                        },
                    )
                )
        elif suffix in {".html", ".htm"}:
            text = _read_html(file_path)
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_path.name,
                            "file_type": "html",
                            "document_name": file_path.name,
                        },
                    )
                )

    return documents


def build_vector_store(
    data_dir: Path = DATA_DIR,
    persist_directory: Path = DB_DIR,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """Construye la base vectorial con fragmentación y persistencia local."""
    load_dotenv(ROOT_DIR / ".env")

    docs = load_documents(data_dir)
    if not docs:
        print(f"No se encontraron documentos en {data_dir}. Agrega archivos PDF, DOCX, HTML, TXT o MD.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Se generaron {len(chunks)} fragmentos con chunk_size={chunk_size} y overlap={chunk_overlap}.")

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
