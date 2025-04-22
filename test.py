from colorama import Fore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

MARKDOWN_FILE_PATH = "data/gradingdoc.md"
QDRANT_COLLECTION_NAME = "qdrant_index"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # "all-MiniLM-L6-v2"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333


def load_or_create_vectore_store(vectore_store=None):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    if not client.collection_exists(collection_name=QDRANT_COLLECTION_NAME):

        loader = UnstructuredMarkdownLoader(MARKDOWN_FILE_PATH)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
        docs = text_splitter.split_documents(documents)
        print(
            Fore.LIGHTBLUE_EX + f"Split document into {len(docs)} chunks." + Fore.RESET
        )

        # Create collection

        client.recreate_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=384, distance=Distance.COSINE
            ),
        )

        vector_store = Qdrant.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name=QDRANT_COLLECTION_NAME,
            host=QDRANT_HOST,
            port=QDRANT_PORT,
        )

        print(
            Fore.LIGHTGREEN_EX
            + f"Saved Qdrant collection: {QDRANT_COLLECTION_NAME}"
            + Fore.RESET
        )
    else:
        vector_store = Qdrant(
            client=client, collection_name=QDRANT_COLLECTION_NAME, embeddings=embeddings
        )
        print(Fore.LIGHTYELLOW_EX + "Loaded existing Qdrant collection." + Fore.RESET)

    return vector_store


def query_documents(query: str) -> str:
    vector_store = load_or_create_vectore_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    results = ""

    print(Fore.YELLOW + f"Found {len(results)} relevant documents." + Fore.RESET)
    for result in retriever.invoke(query):
        results += f"Document: {result.page_content}\n\n"

    return results
    

if __name__ == "__main__":
    vector_store = load_or_create_vectore_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    results = retriever.invoke(
        "what are the eligibilities for the final exams?"
    )

    print(Fore.YELLOW + f"Found {len(results)} relevant documents." + Fore.RESET)
    for result in results:
        print(Fore.CYAN + f"Document: {result.page_content}" + Fore.RESET)
        print(Fore.LIGHTWHITE_EX + f"Metadata: {result.metadata}" + Fore.RESET)
        print(Fore.LIGHTWHITE_EX + "-" * 80 + Fore.RESET)
