from text_utils import load_documents
from embedding_2_database import (
    encode_documents, create_faiss_index, 
    save_index, load_index, 
    save_documents, load_documents_pickle
)
from submit_query import setup_llama_pipeline, get_answer_llama

import os
from sentence_transformers import SentenceTransformer
import torch

def main():
  folder = 'archives-of-nethys-scraper/parsed'
  index_path = "index/faiss_index.index"
  docs_path = "index/documents.pk1"

  # Check if saved index and docs exist
  if os.path.exists(index_path) and os.path.exists(docs_path):
    print("Loading saved index and documents...")
    index = load_index(index_path)
    documents = load_documents_pickle(docs_path)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  else:
    # If no save index or docs exist, Load them and then save them
    print("Loading Documents...")
    documents = load_documents(folder)
    print(f"Loaded {len(documents)} chunks.")
    # Encode documents and build index
    print("Encoding documents and building index...")
    embeddings, model = encode_documents(documents)
    index, device = create_faiss_index(embeddings, use_gpu=False)
    save_documents(documents, docs_path)
    save_index(index, index_path)

  # Hugging Face Token
  print("Connecting to Hugging Face Model...")

  # Load llama
  print("Loading Model...")
  rag_pipeline = setup_llama_pipeline()

  # Query Loop
  while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == 'exit':
      print("Thank you for visiting the Archives! Exiting...")
      break
  
    print("Searching the Archives....")
    answer = get_answer_llama(query, index, model, device, documents, rag_pipeline, k=5)
    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
  main()
