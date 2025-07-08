import streamlit as st
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
import os
from llama_index.llms.ollama import Ollama
from llama_index.core.storage import StorageContext
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from pathlib import Path

load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")

# llm = OpenAI(
#     api_key=os.getenv("OPENAI_API_KEY"),
#     model="gpt-4o-mini"
# )

st.set_page_config(page_title="RAG Institute", page_icon=":book:", layout="wide")

llm = Ollama(
    model="gemma3n:latest"
)

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text"
)


neo4j_vector = Neo4jVectorStore(
    username=os.getenv("NEO4J_USERNAME", ""),
    password=os.getenv("NEO4J_PASSWORD", ""),
    embedding_dimension=768,  # testing with Ollama's embedding dimension
    url=os.getenv("NEO4J_URI", ""),
    instance_id=os.getenv("NEO4J_INSTANCE_ID", ""),
    database="neo4j",
    hybrid_search=True
)

def get_or_create_index():
    """Get existing index or create new one if it doesn't exist"""
    if "index" not in st.session_state:
        storage_context = StorageContext.from_defaults(vector_store=neo4j_vector)
        try:
            # Try to load existing index
            st.session_state["index"] = VectorStoreIndex.from_vector_store(
                neo4j_vector, storage_context=storage_context
            )
        except:
            # If no existing index, create empty one
            st.session_state["index"] = VectorStoreIndex([], storage_context=storage_context)
    
    return st.session_state["index"]


def main():
    st.title("Hello from rag-institute!")
    col1, col2 = st.columns(2)

    with st.sidebar:
        uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:

        # save the doc in `data` folder
        Path("data").mkdir(parents=True, exist_ok=True)
        available_files = os.listdir("data")
        if uploaded_file.name not in available_files: # check if the file is already uploaded
            with open(f"data/{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getbuffer())
        with st.sidebar:
            st.success(f"File {uploaded_file.name} uploaded successfully!")

        # check if the file is already uploaded to Neo4j
        st.session_state["uploaded_files_to_neo4j"] = st.session_state.get(
            "uploaded_files_to_neo4j", []
        )
        # if uploaded_file.name not in st.session_state["uploaded_files_to_neo4j"]:
            # global index
            #load the document
        
        index = get_or_create_index()
        new_doc = SimpleDirectoryReader(input_files=[f"data/{uploaded_file.name}"]).load_data()
        for doc in new_doc:
            index.insert(doc)
        st.session_state["uploaded_files_to_neo4j"].append(uploaded_file.name)

        print(f"File {uploaded_file.name} indexed successfully in Neo4j!")

        with col1:
            query = st.text_input("Ask a question about the uploaded file:")
            if query:
                with st.spinner("Querying the index..."):
                    # Use the LLM to query the index
                    index = get_or_create_index()
                    query_engine = index.as_query_engine(llm=llm,
                                similarity_top_k=10,
                                response_mode="hybrid_summarize")
                    response = query_engine.query(query)
                print(f"Response: {response}")
                st.write("**Response:**")
                st.markdown(response, unsafe_allow_html=True)
        with col2:
            if query:
                with st.spinner("Querying the index..."):
                    # Use the LLM to query the index
                    # index = get_or_create_index()
                    # query_engine = index.as_query_engine(llm=llm,
                    #             similarity_top_k=10,
                    #             response_mode="tree_summarize")
                    
                    # Method 1: Get the retriever to see what documents are retrieved
                    retriever = index.as_retriever(similarity_top_k=10)
                    retrieved_nodes = retriever.retrieve(query)
                    
                    # Display retrieved nodes info
                    st.write(f"**Retrieved {len(retrieved_nodes)} Nodes from Neo4j:**")
                    for i, node in enumerate(retrieved_nodes):
                        st.write(f"**Node {i+1}:**")
                        st.write(f"- **Score:** {node.score}")
                        st.write(f"- **Text:** {node.text[:200]}...")  # First 200 chars
                        st.write(f"- **Metadata:** {node.metadata}")
                        st.write("---")

if __name__ == "__main__":
    main()
    # resp = llm.complete("Who is Paul Graham?")
    # print(resp)
