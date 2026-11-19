import os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Import our loader from Step 1
from step1_loader import load_and_chunk

# Load environment variables from .env
load_dotenv()

# 1. Define the strict schema for our Knowledge Graph triples
class Relationship(BaseModel):
    source: str = Field(description="Name of the source entity (e.g., 'TechNova Corp')")
    source_type: str = Field(description="Type/Label of source entity (e.g., 'Company', 'Person', 'Location', 'Product')")
    relation: str = Field(description="Relationship in UPPER_SNAKE_CASE (e.g., 'FOUNDED_BY', 'HEADQUARTERED_IN')")
    target: str = Field(description="Name of the target entity (e.g., 'Elena Rostova')")
    target_type: str = Field(description="Type/Label of target entity (e.g., 'Company', 'Person', 'Location', 'Product')")

class KnowledgeGraph(BaseModel):
    relationships: List[Relationship] = Field(description="List of extracted entity relationships")

# 2. Initialize the LLM and bind the structured output schema
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(KnowledgeGraph)

# 3. Create the extraction prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a precise Knowledge Graph extraction engine. "
        "Extract entities and their relationships from the provided text. "
        "Rules:\n"
        "- Normalize entity names (e.g., use 'TechNova Corp' consistently instead of 'TechNova' or 'The company').\n"
        "- Use clear, concise UPPER_SNAKE_CASE for relationship types.\n"
        "- Only extract facts explicitly stated in the text."
    )),
    ("human", "Extract the knowledge graph triples from this text:\n\n{text}")
])

extraction_chain = prompt | structured_llm

def extract_graph_from_chunks(chunks) -> List[Relationship]:
    all_relationships = []
    
    for i, chunk in enumerate(chunks):
        print(f"Processing Chunk {i+1}/{len(chunks)}...")
        result: KnowledgeGraph = extraction_chain.invoke({"text": chunk.page_content})
        all_relationships.extend(result.relationships)
        
    return all_relationships

if __name__ == "__main__":
    # Load chunks from Step 1
    chunks = load_and_chunk("sample_data.txt")
    
    # Extract triples
    triples = extract_graph_from_chunks(chunks)
    
    print("\n=== Extracted Knowledge Graph Triples ===")
    for t in triples:
        print(f"({t.source} [{t.source_type}]) -[:{t.relation}]-> ({t.target} [{t.target_type}])")