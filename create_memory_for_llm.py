"""we are here to build a multi-document FAISS knowledge base."""

import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# configuration


from config import (
    DATA_PATH,
    FAISS_INDEX_PATH as DB_FAISS_PATH,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    BATCH_SIZE,
)

# ==========================================
# loads the multiple pdf documents from the data directory and extracts text pages
# ==========================================


def load_pdf_files():

    pdf_files = sorted(DATA_PATH.glob("*.pdf"))

    if not pdf_files:

        raise FileNotFoundError("No PDF files found in the data directory.")

    documents = []
    failed_files = []

    print("\n" + "=" * 55)
    print("MEDICALLY AI - KNOWLEDGE BASE BUILDER")
    print("=" * 55)

    print(f"\nPDF files found: {len(pdf_files)}\n")

    for i, pdf_file in enumerate(pdf_files, 1):

        try:

            print(f"[{i}/{len(pdf_files)}] " f"Processing: {pdf_file.name}")

            loader = PyPDFLoader(str(pdf_file))

            pages = [page for page in loader.load() if page.page_content.strip()]

            if not pages:
                raise ValueError("No extractable text")

            documents.extend(pages)

            print(f"    Text pages loaded: {len(pages)}")

        except Exception as error:

            failed_files.append(pdf_file.name)

            print(f"    Failed: {error}")

    if failed_files:

        raise RuntimeError(
            "Indexing aborted. Fix or remove failed PDFs: " + ", ".join(failed_files)
        )

    if not documents:

        raise ValueError("No readable text was extracted.")

    print(f"\nTotal text pages: {len(documents)}")

    return documents


# Document chunking step


def create_chunks(documents):

    print("\nSplitting documents into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks created.")

    print(f"Chunk size: {CHUNK_SIZE} characters")

    print(f"Chunk overlap: {CHUNK_OVERLAP} characters")

    print(f"Total chunks: {len(chunks)}")

    return chunks


# Embedding model step


def get_embedding_model():

    print("\nLoading MiniLM embedding model...")

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": BATCH_SIZE},
        show_progress=True,
    )


# build the faiss index


def main():

    start_time = time.perf_counter()

    documents = load_pdf_files()

    chunks = create_chunks(documents)

    embedding_model = get_embedding_model()

    print("\nGenerating vector embeddings...")
    print("Please monitor the progress bar below.\n")

    database = FAISS.from_documents(chunks, embedding_model)

    DB_FAISS_PATH.mkdir(parents=True, exist_ok=True)

    database.save_local(str(DB_FAISS_PATH))

    elapsed = time.perf_counter() - start_time

    print("\n" + "=" * 55)
    print("KNOWLEDGE BASE CREATED SUCCESSFULLY")
    print("=" * 55)

    print(f"Text pages       : {len(documents)}")
    print(f"Text chunks      : {len(chunks)}")
    print(f"Indexed vectors  : {database.index.ntotal}")
    print(f"Vector dimension : {database.index.d}")
    print(f"Processing time  : {elapsed:.2f} seconds")
    print(f"Database path    : {DB_FAISS_PATH}")

    print("=" * 55)


if __name__ == "__main__":

    main()
