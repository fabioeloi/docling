from typing import Iterator, Union, List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import pipeline

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument

from docling.document_converter import DocumentConverter

# Custom loader for Docling documents
class DoclingPDFLoader(BaseLoader):
    def __init__(self, file_path: Union[str, List[str]]) -> None:
        self._file_paths = file_path if isinstance(file_path, list) else [file_path]
        self._converter = DocumentConverter()

    def lazy_load(self) -> Iterator[LCDocument]:
        for source in self._file_paths:
            dl_doc = self._converter.convert(source).document
            text = dl_doc.export_to_markdown()
            yield LCDocument(page_content=text)

def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into chunks with overlap."""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        # If this is not the first chunk, back up to include overlap
        if start > 0:
            start = start - overlap
        # If this is not the last chunk, try to break at a period or newline
        if end < text_len:
            # Look for the last period or newline in the chunk
            last_period = text.rfind('.', start, end)
            last_newline = text.rfind('\n', start, end)
            break_point = max(last_period, last_newline)
            if break_point > start:
                end = break_point + 1
        
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        start = end
    
    return chunks

def find_relevant_chunks(query: str, chunks: List[str], embeddings: np.ndarray, model: SentenceTransformer, top_k: int = 2) -> List[Tuple[str, float]]:
    """Find the most relevant chunks for a query using cosine similarity."""
    # Generate embedding for the query
    query_embedding = model.encode([query])[0]
    
    # Calculate cosine similarity between query and all chunks
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    
    # Get indices of top_k most similar chunks
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Return chunks and their similarity scores
    return [(chunks[i], similarities[i]) for i in top_indices]

def answer_question(query: str, chunks: List[str], embeddings: np.ndarray, 
                   embedding_model: SentenceTransformer, text_generator) -> str:
    """Answer a question using relevant chunks as context and an LLM."""
    relevant_chunks = find_relevant_chunks(query, chunks, embeddings, embedding_model)
    
    # Format the context from relevant chunks
    context = "\n".join(f"{chunk}" for chunk, score in relevant_chunks)
    
    # Generate answer using the pipeline
    prompt = f"""Based on the following context, please answer the question. 
If the answer cannot be found in the context, say "I cannot answer this based on the given context."

Context:
{context}

Question: {query}

Answer: """
    
    result = text_generator(prompt, max_new_tokens=256, temperature=0.7, top_p=0.9)
    answer = result[0]['generated_text'][len(prompt):]
    
    return (f"Based on the following relevant passages:\n\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer: {answer}")

def main():
    # Initialize components
    pdf_path = "tests/data/2305.03393v1-pg9.pdf"  # Using a smaller PDF for testing
    loader = DoclingPDFLoader(pdf_path)
    
    print("Loading and processing document...")
    # Load and process documents
    documents = list(loader.lazy_load())
    
    # Initialize the embedding model
    print("\nInitializing embedding model...")
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Initialize the text generation pipeline
    print("\nInitializing language model...")
    text_generator = pipeline(
        'text-generation',
        model='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        torch_dtype=torch.float16,
        device_map='auto'
    )
    
    for doc in documents:
        print("\nOriginal document content:")
        print("-" * 50)
        print(doc.page_content[:500])
        print("-" * 50)
        
        # Split the document into chunks
        print("\nSplitting document into chunks...")
        chunks = split_text(doc.page_content)
        print(f"Created {len(chunks)} chunks")
        
        # Generate embeddings for each chunk
        print("\nGenerating embeddings...")
        embeddings = embedding_model.encode(chunks)
        
        # Try some example questions
        print("\nAnswering questions about the document:")
        questions = [
            "What dataset was used for HPO and why was it chosen?",
            "What CPU was used for the experiments and how was it configured?",
            "What are the advantages of OTSL over HTML according to the results?",
        ]
        
        for question in questions:
            print("\n" + "="*50)
            print(answer_question(
                question, 
                chunks, 
                embeddings, 
                embedding_model,
                text_generator
            ))

if __name__ == "__main__":
    main()
