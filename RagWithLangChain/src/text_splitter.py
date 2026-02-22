from langchain_text_splitters import RecursiveCharacterTextSplitter
from document_loader import docs
from config import CHUNK_SIZE, CHUNK_OVERLAP

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(len(all_splits))
print(docs[1].metadata)