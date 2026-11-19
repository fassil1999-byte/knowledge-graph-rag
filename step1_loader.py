from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk(file_path: str):
    # 1. Load the document
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s).")

    # 2. Split into chunks
    # Using a small chunk size here just to demonstrate splitting on our short text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=40,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"Split into {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ---")
        print(chunk.page_content)
        print()
        
    return chunks

if __name__ == "__main__":
    chunks = load_and_chunk("sample_data.txt")