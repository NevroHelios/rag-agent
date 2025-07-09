from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_neo4j.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_community.vectorstores import Neo4jVector



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
            value = value.strip()
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
    """Identifying information about entities in the text."""

    address: str = Field(description="An address, such as a street address or a location.")
    attribute: str = Field(description="An attribute or characteristic of an entity.")
    author: str = Field(description="A single author of the paper or a cited work.")
    authors: str = Field(description="List of authors associated with the paper.")
    chunk: str = Field(description="A section or meaningful unit of the text.")
    company: str = Field(description="A company or corporate entity mentioned in the text.")
    concept: str = Field(description="An abstract idea, principle, or domain-specific notion.")
    curve: str = Field(description="A graphical representation or mathematical curve described in the paper.")
    cycle: str = Field(description="A repeated process or time cycle referenced in the study.")
    data: str = Field(description="Specific data values, raw or processed, mentioned in the paper.")
    dataset: str = Field(description="A collection of data used, referenced, or produced in the study.")
    department: str = Field(description="A specific department within an organization or institution.")
    distribution: str = Field(description="The statistical or spatial distribution of data or variables.")
    document: str = Field(description="A referenced or related document, article, or paper.")
    electric_current: str = Field(description="Mentions of electric current or related electrical measurements.")
    entity: str = Field(description="A general named entity not otherwise categorized.")
    environment: str = Field(description="The environmental context or conditions mentioned.")
    equation: str = Field(description="A mathematical equation or formula stated in the text.")
    event: str = Field(description="An occurrence or happening described in the context of the study.")
    figure: str = Field(description="A referenced figure, chart, or diagram within the paper.")
    function: str = Field(description="A mathematical or computational function described or used.")
    group: str = Field(description="A collection of people, items, or elements considered together.")
    index: str = Field(description="An index value or indexing term used in the paper.")
    instrument: str = Field(description="A scientific or technical instrument used in data collection.")
    journal: str = Field(description="The journal where the paper or referenced articles are published.")
    location: str = Field(description="A geographic location or place mentioned.")
    magnetometer: str = Field(description="A specific instrument measuring magnetic fields.")
    measurement: str = Field(description="The act or result of measuring a quantity.")
    number: str = Field(description="A numerical value mentioned in the text.")
    orcid_id: str = Field(description="The ORCID identifier for an author.")
    organization: str = Field(description="An organization or institution involved or referenced.")
    person: str = Field(description="A named individual mentioned in the text.")
    phenomenon: str = Field(description="A scientific or observable phenomenon discussed.")
    planet: str = Field(description="A planet mentioned in the astronomical or environmental context.")
    publication: str = Field(description="A published work referenced or discussed.")
    quantity: str = Field(description="A measurable amount or value.")
    reference: str = Field(description="A citation or bibliographic reference.")
    region: str = Field(description="A specific area or region, geographic or conceptual.")
    resource: str = Field(description="Any resource—material, computational, or informational—mentioned.")
    satellite: str = Field(description="A satellite referenced in context to data collection or observation.")
    spacecraft: str = Field(description="A spacecraft or probe mentioned in the study.")
    state: str = Field(description="A physical or logical state or condition of a system or material.")
    structure: str = Field(description="A physical, logical, or organizational structure.")
    system: str = Field(description="A system—technical, natural, or conceptual—described in the paper.")
    thresholds: str = Field(description="Threshold values or limits defined or measured.")
    time_unit: str = Field(description="Units of time used in measurements or descriptions.")
    tool: str = Field(description="A software or hardware tool used in the analysis or study.")
    unit_of_measurement: str = Field(description="A standardized unit for measuring variables.")
    value: str = Field(description="A specific value, usually numerical or categorical, relevant to the context.")
    variable: str = Field(description="A changing or measured quantity in the study.")
    website: str = Field(description="A URL or web-based resource referenced.")
    year: str = Field(description="A specific year mentioned in the paper.")