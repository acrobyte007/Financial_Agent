
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma


model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceEmbeddings(
model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)   


vector_store = Chroma(
collection_name="collection",
embedding_function=hf,
persist_directory="chroma_langchain_db",
)


def embed_chunks(chunks: list[str]) -> None:
    """
    Embeds a list of text chunks and stores them in the Chroma vector store.
    
    Args:
        chunks (list[str]): List of text chunks to embed
        doc_id (str): Document identifier
        user_id (str): User identifier
    """
    for i, chunk in enumerate(chunks):
        
        document = Document(
            page_content=chunk,
        )
        vector_store.add_documents([document])

def get_chunks(query:str):
    results = vector_store.similarity_search(
    query,
    k=5
)
    list_of_chunks=[]
    for res in results:
        list_of_chunks.append(res.page_content)
    return list_of_chunks
