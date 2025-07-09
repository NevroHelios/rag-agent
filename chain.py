from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

from utils import GraphRetriever, Entities
from config import CFG

# load the env variables
load_dotenv()

def build_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert in scientific papers. Your task is to extract entities from the text. "
            ),
            (
                "human",
                "Use the given format to extract information from the following"
                "input: {question}"
            )
        ]
    )

    entity_chain = prompt | CFG.llm.with_structured_output(Entities)

    # retriever
    retriever = GraphRetriever(
        graph=CFG.graph,
        entity_chain=entity_chain,
        embedding_model=CFG.embedding_model
    )

    #chain
    template = """Answer the question based on the provided data.
    Context: 
    {context}

    Question: {question}

    Use the context to answer the question as accurately as possible. Be concise and to the point.

    Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {
            "context": retriever.full_retriever,
            "question": RunnablePassthrough(),
        } 
        | prompt
        | CFG.llm
        | StrOutputParser()
    )

    return chain