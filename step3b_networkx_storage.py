import json
from typing import List
import networkx as nx

# Import functions from previous steps
from step1_loader import load_and_chunk
from step2_extractor import extract_graph_from_chunks, Relationship


class NetworkXGraphStore:
    def __init__(self):
        # MultiDiGraph allows directed edges and multiple relationships between the same two nodes
        self.graph = nx.MultiDiGraph()

    def ingest_triples(self, triples: List[Relationship]):
        for triple in triples:
            # Add source and target nodes with their entity type as an attribute
            self.graph.add_node(triple.source, type=triple.source_type)
            self.graph.add_node(triple.target, type=triple.target_type)

            # Avoid adding duplicate identical edges (same source, target, and relation)
            existing_edges = self.graph.get_edge_data(triple.source, triple.target) or {}
            already_exists = any(
                edge_attrs.get("relation") == triple.relation
                for edge_attrs in existing_edges.values()
            )

            if not already_exists:
                self.graph.add_edge(
                    triple.source,
                    triple.target,
                    relation=triple.relation
                )

        print(
            f"Built NetworkX graph with {self.graph.number_of_nodes()} nodes "
            f"and {self.graph.number_of_edges()} edges."
        )

    def save_to_json(self, filepath: str = "knowledge_graph.json"):
        """Persist the in-memory graph to a JSON file."""
        data = nx.node_link_data(self.graph)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Saved NetworkX graph to {filepath}")

    @staticmethod
    def load_from_json(filepath: str = "knowledge_graph.json") -> nx.MultiDiGraph:
        """Reload a persisted NetworkX graph from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return nx.node_link_graph(data)


if __name__ == "__main__":
    # 1. Load and extract triples (Steps 1 & 2)
    chunks = load_and_chunk("sample_data.txt")
    triples = extract_graph_from_chunks(chunks)

    # 2. Build and save the NetworkX graph
    nx_store = NetworkXGraphStore()
    nx_store.ingest_triples(triples)
    nx_store.save_to_json("knowledge_graph.json")

    # 3. Verify by reloading and printing stored edges
    loaded_graph = NetworkXGraphStore.load_from_json("knowledge_graph.json")
    print("\n=== Verified Edges in NetworkX ===")
    for u, v, attrs in loaded_graph.edges(data=True):
        u_type = loaded_graph.nodes[u].get("type", "Entity")
        v_type = loaded_graph.nodes[v].get("type", "Entity")
        print(f"({u} [{u_type}]) -[:{attrs['relation']}]-> ({v} [{v_type}])")