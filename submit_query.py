# Set up llama model for the rag by setting up/defining the model and then making the get_answer_llama module.
from transformers import pipeline
import torch
from huggingface_hub import login
import numpy as np
import os

def setup_llama_pipeline():
  token = os.getenv("HF_TOKEN")
  if not token:
    raise ValueError("Hugging Face Token not found. Please set the HF_TOKEN enviroment variable")
  login(token)
  dtype = torch.float16 if torch.cuda.is_available() else torch.float32
  return pipeline(model="meta-llama/Llama-3.2-1B", torch_dtype=dtype, device_map="auto")

def get_answer_llama(question, faiss_index, embed_model, device, documents, rag_pipeline, k=3):
  # encode the query and normalize it for cosine simlairty
  query_embedding = embed_model.encode(question, convert_to_tensor=True).cpu().numpy()
  query_embedding = query_embedding / np.linalg.norm(query_embedding)  # normalize
  query_embedding = query_embedding.reshape(1, -1)

  # Search the Index
  D, I = faiss_index.search(query_embedding, k)
  retrieved_documents = [documents[i] for i in I[0]]

  # Build Prompt
  context = " ".join([str(doc) for doc in retrieved_documents])
  prompt = (
    "Answer the following question based only on the provided context. "
    "If the context contains a description of a rule, explain it briefly and accurately. "
    "If the answer is not in the context, say 'I don't know based on context' \n\n"
    f"Context: {context}\nQuestion: {question}\nAnswer:"
  )
  #f"Answer the following question based only on the provided context. If the answer is not in the context, say 'I don't know based on the context.\n\nContext: {context}\nQuestion: {question}\nAnswer:"

  # Generate response with llama
  output = rag_pipeline(prompt, max_new_tokens=100, do_sample=False)

  return output[0]['generated_text']
