from langchain_community.document_loaders import PyPDFLoader
from config import DATA_DIR

file_path =DATA_DIR / "nke-10k-2023.pdf"


def load_documents(file_name="nke-10k-2023.pdf"):
    """Loads a PDF from the data directory and returns the documents."""
    file_path = DATA_DIR / file_name
    loader = PyPDFLoader(str(file_path))
    return loader.load()


if __name__ == "__main__":
    # This only runs if you test this file directly
    docs = load_documents()
    print(f"Loaded {len(docs)} documents.")