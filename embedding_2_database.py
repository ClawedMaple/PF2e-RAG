import numpy as np
import torch
import faiss
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
import pickle
import os

def setup_device():
  return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def l2_normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def encode_documents(documents, model_name='all-MiniLM-L6-v2', batch_size=64):

  model = SentenceTransformer(model_name)

  # Encode documents into embeddings in batches
  document_embeddings = []
  for i in tqdm(range(0, len(documents), batch_size), desc="Encoding Documents"):
      batch = documents[i : i + batch_size]

      # Convert list-like objects to strings
      processed_batch = []
      for item in batch:
          if isinstance(item, str):
              processed_batch.append(item)  # Keep strings as they are
          elif isinstance(item, list):
              # If item is a list, ensure all elements are strings
              converted_items = [str(sub_item) for sub_item in item]
              processed_batch.append(' '.join(converted_items))  # Join list items into a single string
          else:
              # Handle other data types if necessary (e.g., convert to strings)
              processed_batch.append(str(item))

      # If batch is empty after processing, skip
      if not processed_batch:
          continue

      # Encode the processed batch
      batch_embeddings = model.encode(processed_batch, convert_to_tensor=True)
      document_embeddings.extend(batch_embeddings.cpu())

  # Convert to numpy array after moving all tensors to CPU
  document_embeddings = np.array([tensor.numpy() for tensor in document_embeddings])
  document_embeddings = l2_normalize(document_embeddings)

  return document_embeddings, model

def create_faiss_index(embeddings, use_gpu=False):
  dim = embeddings.shape[1]

  # Use inner product (dot product) index with normalized vectors = cosine similarity
  index = faiss.IndexFlatIP(dim)

  index.add(embeddings)

  device = setup_device()
  return index, device

def save_index(index, path="index/faiss_index.index"):
   faiss.write_index(index, path)
   print(f"FAISS index saved to {path}")

def load_index(path="index/faiss_index.index"):
   if os.path.exists(path):
      print(f"Loaded FAISS index from {path}")
      return faiss.read_index(path)
   else:
      print("Index file not found.")
      return None

def save_documents(documents, path="index/documents.pk1"):
   with open(path, "wb") as f:
      pickle.dump(documents, f)
      print(f"Saved documents to {path}")

def load_documents_pickle(path="index/documents.pk1"):
   if os.path.exists(path):
      with open(path, "rb") as f:
         print(f"Loaded documents from {path}")
         return pickle.load(f)
   else:
      print("Documents file not found")
      return None

      
