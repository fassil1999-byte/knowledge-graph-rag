import os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from neo4j import GraphDatabase

# Reuse our NetworkX loader from Step 3B
from step3b_networkx_storage import NetworkXGraphStore

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# --- 1. Extract Entities from the User's Question ---
class QueryEntities(BaseModel):
    entities: List[str] = Field(
        description="Key entity names mentioned in the question (e.g., 'TechNova Corp', 'Marcus Vance')"
    )


entity_extractor = llm.with_structured_output(QueryEntities)


def extract_question_entities(question: str) -> List[str]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract the main named entities (companies, people, products, locations) from the user's question."),
        ("human", "{question}")
    ])
    result = (prompt | entity_extractor).invoke({"question": question})
    return result.entities


# --- 2A. Retrieve 2-Hop Context from Neo4j AuraDB ---
def retrieve_from_neo4j(entities: List[str]) -> List[str]:
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    triples_context = set()

    # Match 1-hop and optional 2-hop paths from any node matching the extracted entity names
    cypher = """
    MATCH (start)
    WHERE toLower(start.name) CONTAINS toLower($entity)
    MATCH (start)-[r1]-(mid)
    OPTIONAL MATCH (mid)-[r2]-(end)
    RETURN start.name AS s1, type(r1) AS rel1, mid.name AS t1,
           mid.name AS s2, type(r2) AS rel2, end.name AS t2
    """

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        for entity in entities:
            records, _, _ = driver.execute_query(cypher, entity=entity)
            for rec in records:
                if rec["s1"] and rec["rel1"] and rec["t1"]:
                    triples_context.add(f"({rec['s1']}) -[:{rec['rel1']}]-> ({rec['t1']})")
                if rec["s2"] and rec["rel2"] and rec["t2"]:
                    triples_context.add(f"({rec['s2']}) -[:{rec['rel2']}]-> ({rec['t2']})")

    return sorted(triples_context)


# --- 2B. Retrieve 2-Hop Context from NetworkX ---
def retrieve_from_networkx(entities: List[str], filepath: str = "knowledge_graph.json") -> List[str]:
    graph = NetworkXGraphStore.load_from_json(filepath)
    triples_context = set()

    # Find matching seed nodes (case-insensitive substring match)
    seed_nodes = [
        node for node in graph.nodes()
        if any(ent.lower() in node.lower() for ent in entities)
    ]

    # Traverse 2 hops (both outgoing and incoming edges)
    visited = set()
    frontier = set(seed_nodes)

    for _hop in range(2):
        next_frontier = set()
        for node in frontier:
            if node in visited:
                continue
            visited.add(node)

            # Outgoing edges: node -> neighbor
            for _, target, attrs in graph.out_edges(node, data=True):
                triples_context.add(f"({node}) -[:{attrs['relation']}]-> ({target})")
                next_frontier.add(target)

            # Incoming edges: neighbor -> node
            for source, _, attrs in graph.in_edges(node, data=True):
                triples_context.add(f"({source}) -[:{attrs['relation']}]-> ({node})")
                next_frontier.add(source)

        frontier = next_frontier

    return sorted(triples_context)


# --- 3. Generate Final RAG Answer ---
def generate_rag_answer(question: str, graph_context: List[str]) -> str:
    if not graph_context:
        return "No relevant information found in the Knowledge Graph."

    formatted_context = "\n".join(graph_context)
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a Knowledge Graph RAG assistant. Answer the user's question using ONLY "
            "the retrieved graph relationships below. If the answer is not in the graph context, say so.\n\n"
            "Retrieved Graph Context:\n{context}"
        )),
        ("human", "{question}")
    ])

    response = (prompt | llm).invoke({
        "context": formatted_context,
        "question": question
    })
    return response.content


if __name__ == "__main__":
    # A multi-hop question that requires connecting Marcus Vance -> TechNova -> DataCore Labs -> Seattle
    question = "Who is the CTO of TechNova Corp, where did they work previously, and where is that company based?"

    print(f"Question: {question}\n")
    entities = extract_question_entities(question)
    print(f"Extracted Question Entities: {entities}\n")

    # --- Test 1: Neo4j AuraDB ---
    print("=== 1. Neo4j AuraDB Graph RAG ===")
    neo4j_context = retrieve_from_neo4j(entities)
    print("Retrieved Triples:")
    for triple in neo4j_context:
        print("  ", triple)
    print("\nNeo4j Answer:", generate_rag_answer(question, neo4j_context))

    print("\n" + "=" * 50 + "\n")

    # --- Test 2: NetworkX ---
    print("=== 2. NetworkX Graph RAG ===")
    nx_context = retrieve_from_networkx(entities)
    print("Retrieved Triples:")
    for triple in nx_context:
        print("  ", triple)
    print("\nNetworkX Answer:", generate_rag_answer(question, nx_context))