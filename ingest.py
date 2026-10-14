from pathlib import Path
import shutil

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

DATA_DIR = Path("data")
DB_DIR = Path("chroma_db")
EMBEDDING_MODEL = "nomic-embed-text"


def load_text_documents():
    """Load each .txt file and retain its filename as source metadata."""
    documents = []
    for file_path in sorted(DATA_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()
        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": file_path.name}
                )
            )
    return documents


def main():
    documents = load_text_documents()
    if not documents:
        raise SystemExit("No .txt files found in the data folder.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=60
    )
    chunks = splitter.split_documents(documents)

    # Rebuild the local index so repeated runs do not duplicate old chunks.
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
        collection_name="company_policies"
    )

    print(f"Loaded {len(documents)} documents.")
    print(f"Created {len(chunks)} chunks.")
    print(f"Saved the vector database to: {DB_DIR.resolve()}")


if __name__ == "__main__":
    main()