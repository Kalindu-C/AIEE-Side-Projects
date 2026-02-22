from langchain_openai import OpenAIEmbeddings
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from config import EMBEDDING_MODEL_NAME, VECTOR_SIZE

def get_embeddings():
    """Initialize and return the embedding model."""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)


def get_vector_store(collection_name="my_documents", documents=None):

    """
    Initializes the Qdrant client and returns the VectorStore.
    If 'documents' are provided, it adds them to the store.
    """

    print("Embedding docs...")
    embeddings = get_embeddings()

    # 1. Initialize Client
    client = QdrantClient(":memory:") # Replace with QDRANT_URL from config

    # 2. Create collection if it doesn't exist
    if not client.collection_exists(collection_name):
        # It's better to know your vector size than to make an API call to find out
        # But if you must use the hack, do it here so it only runs ONCE when creating the collection
        vector_size = len(embeddings.embed_query("sample text")) 
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

    # 3. Create the LangChain VectorStore wrapper
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )

    # 4. Add documents if provided (Ingestion mode)
    if documents:
        print(f"Adding {len(documents)} documents to Qdrant...")
        vector_store.add_documents(documents=documents)
        
    return vector_store

if __name__ == "__main__":
    # Now you control exactly when the loading and splitting happens!
    from document_loader import load_documents
    from text_splitter import split_documents
    
    print("Loading docs...")
    docs = load_documents()
    
    print("Splitting docs...")
    all_splits = split_documents(docs)
    

    
    print("Initializing vector store and adding documents...")
    # We pass the splits directly into the function
    vector_store = get_vector_store(collection_name="test", documents=all_splits)
    
    print("Done! Vector store is ready.")