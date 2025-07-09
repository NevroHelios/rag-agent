from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_neo4j.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_community.vectorstores import Neo4jVector
from typing import List


class GraphRetriever:
    """To retrieve relevant documents and entities from a Neo4j graph database."""
    
    def __init__(self, graph, entity_chain, embedding_model):
        self.graph = graph
        self.entity_chain = entity_chain
        self.embedding_model = embedding_model

    def _generate_full_text_query(self, input: str) ->  str:
        words = [el for el in remove_lucene_chars(input).split() if el]
        if not words:
            return ""
        full_text_query = " AND ".join([f"{word}~2" for word in words])
        return full_text_query.strip()

    def _graph_retriever(self, question: str) -> str:
        """Retrieves relevant documents and related entities based on the question."""
        result = ""
        entities = self.entity_chain.invoke({"question": question}) 
        seen = set()
        for entity_type, value in entities.dict().items():
            if not value:
                continue
            value = value[0].strip()
            if not value or value in seen:
                continue  # Skip empty fields
            seen.add(value)

            fuzzy_query = self._generate_full_text_query(value)

            response = self.graph.query(
                """
                CALL db.index.fulltext.queryNodes('entityFullTextIndex', $query, {limit: 5})
                YIELD node, score
                CALL {
                    WITH node
                    OPTIONAL MATCH (node)-[:MENTIONS|RELATED_TO|CITED_IN|GENERATED_BY*1..2]-(doc:Document)
                    RETURN DISTINCT doc.title AS title, doc.abstract AS abstract
                }
                RETURN title, abstract, score
                ORDER BY score DESC
                LIMIT 3
                """,
                {"query": fuzzy_query}
            )

            for record in response:
                result += f"Matched Field: {entity_type}\n"
                result += f"Entity: {value}\n"
                result += f"Title: {record['title']}\n"
                result += f"Abstract: {record['abstract']}\n"
                result += f"Score: {record['score']:.2f}\n\n"

        return result.strip()
    
    def _vector_retriever(self):
        vector_index = Neo4jVector.from_existing_graph(
            embedding=self.embedding_model,
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text", "source", "title"],
            embedding_node_property="embedding"
        )
        return vector_index.as_retriever()

    def full_retriever(self, query: str):
        """Retrieves relevant documents and entities based on the query."""
        graph_data = self._graph_retriever(question=query)
        vector_data = [el.model_dump()["page_content"] for el in self._vector_retriever().invoke(query)]

        final_data = f"""Graph Data:\n{graph_data}\n\nVector Data:\n
                    {"\nDocument".join(vector_data)}"""
        
        return final_data.strip()


class Entities(BaseModel):
    """A generalized model for identifying key entities in various documents,
    such as research papers and academic syllabi."""

    person: List[str] = Field(
        description="Names of individuals, such as authors, professors, or cited researchers.",
        default_factory=list
    )
    organization: List[str] = Field(
        description="Organizations, including universities, departments, research institutes, or companies.",
        default_factory=list
    )
    topic: List[str] = Field(
        description="Key subjects, concepts, or topics discussed in the text.",
        default_factory=list
    )
    publication: List[str] = Field(
        description="Cited or mentioned publications, such as papers, books, or articles.",
        default_factory=list
    )
    location: List[str] = Field(
        description="Geographical or institutional locations mentioned.",
        default_factory=list
    )
    date: List[str] = Field(
        description="Specific dates or years relevant to the document's context.",
        default_factory=list
    )
    course: List[str] = Field(
        description="The names or codes of academic courses, if mentioned.",
        default_factory=list
    )
    assessment: List[str] = Field(
        description="Methods of evaluation, such as exams, quizzes, projects, or assignments.",
        default_factory=list
    )