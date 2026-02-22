from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

def split_documents(docs):
    """Takes a list of documents and returns chunked splits."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP, 
        add_start_index=True)

    return text_splitter.split_documents(docs)


if __name__ == "__main__":
    # Testing code
    from document_loader import load_documents
    docs = load_documents()
    splits = split_documents(docs)
    print(f"Created {len(splits)} splits.")