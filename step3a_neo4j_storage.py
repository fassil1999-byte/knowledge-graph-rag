import os
import re
from typing import List
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Import functions from previous steps
from step1_loader import load_and_chunk
from step2_extractor import extract_graph_from_chunks, Relationship

load_dotenv()

def sanitize_label(label: str) -> str:
    """Ensure labels/relationships only contain alphanumeric chars and underscores."""
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", label.strip())
    return cleaned if cleaned else "Entity"

class Neo4jGraphStore:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        # Verify connection immediately
        self.driver.verify_connectivity()
        print("Connected to Neo4j AuraDB successfully!")

    def close(self):
        self.driver.close()

    def ingest_triples(self, triples: List[Relationship]):
        with self.driver.session() as session:
            for triple in triples:
                src_label = sanitize_label(triple.source_type)
                tgt_label = sanitize_label(triple.target_type)
                rel_type = sanitize_label(triple.relation).upper()

                # MERGE prevents duplicate nodes and relationships
                cypher_query = f"""
                MERGE (a:{src_label} {{name: $source_name}})
                MERGE (b:{tgt_label} {{name: $target_name}})
                MERGE (a)-[r:{rel_type}]->(b)
                """
                
                session.run(
                    cypher_query,
                    source_name=triple.source,
                    target_name=triple.target
                )
        print(f"Ingested {len(triples)} triples into Neo4j AuraDB.")

if __name__ == "__main__":
    # 1. Run Step 1 & Step 2 to get our triples
    chunks = load_and_chunk("sample_data.txt")
    triples = extract_graph_from_chunks(chunks)

    # 2. Initialize Neo4j connection
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    store = Neo4jGraphStore(uri, user, password)
    try:
        # 3. Store the graph
        store.ingest_triples(triples)
    finally:
        store.close()