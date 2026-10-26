import json
from pathlib import Path

import faiss
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_DIR = Path(__file__).resolve().parent
PDF_FOLDER = PROJECT_DIR / "data"
VECTORSTORE_DIR = PROJECT_DIR / "vectorstore"
INDEX_PATH = VECTORSTORE_DIR / "index.faiss"
DOCUMENTS_PATH = VECTORSTORE_DIR / "documents.json"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    if not PDF_FOLDER.is_dir():
        raise FileNotFoundError(f"PDF folder not found: {PDF_FOLDER}")

    pdf_paths = sorted(PDF_FOLDER.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {PDF_FOLDER}")

    documents = []
    for pdf_path in pdf_paths:
        print(f"Loading: {pdf_path.name}")
        documents.extend(PyPDFLoader(str(pdf_path)).load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    split_documents = splitter.split_documents(documents)
    if not split_documents:
        raise ValueError(
            "No text was extracted from the PDFs; check that they contain "
            "readable text."
        )

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectors = np.asarray(
        embedding_model.embed_documents(
            [document.page_content for document in split_documents]
        ),
        dtype=np.float32,
    )
    if vectors.ndim != 2 or vectors.shape[0] != len(split_documents):
        raise ValueError("The embedding model returned an invalid vector array.")
    if not np.isfinite(vectors).all():
        raise ValueError("The embedding model returned non-finite vector values.")

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    safe_chunks = [
        {
            "text": document.page_content,
            "source": Path(str(document.metadata.get("source", ""))).name,
            "page": int(document.metadata.get("page", 0)),
        }
        for document in split_documents
    ]
    payload = {
        "version": 1,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "chunks": safe_chunks,
    }

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    temporary_index = INDEX_PATH.with_name("index.faiss.tmp")
    temporary_documents = DOCUMENTS_PATH.with_name("documents.json.tmp")
    faiss.write_index(index, str(temporary_index))
    temporary_documents.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_index.replace(INDEX_PATH)
    temporary_documents.replace(DOCUMENTS_PATH)

    print(f"Loaded {len(documents)} PDF pages and created {len(safe_chunks)} chunks.")
    print(f"College knowledge base saved to {VECTORSTORE_DIR}")


if __name__ == "__main__":
    main()
