from langchain_community.document_loaders import PyPDFLoader
from config import DATA_DIR

file_path =DATA_DIR / "nke-10k-2023.pdf"
print(file_path)

loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs))

print(f"{docs[0].page_content[:200]}\n")
print(docs[1].metadata)