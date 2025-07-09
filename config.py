import os
from langchain.graphs.neo4j_graph import Neo4jGraph
from langchain_ollama.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()

class CFG:
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        database="neo4j",
    )

    # llm to talk to client
    llm = ChatOllama(
        model="llama3.1:8b",
        temperature=0,
    )

    # llm = ChatOpenAI(
    #     model="gpt-4o-mini",
    #     temperature=0,
    #     max_tokens=1000,
    #     streaming=True,
    # )

    # embedding model
    embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")
    