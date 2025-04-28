import os
import json
import re

def extract_text(data):
  extracted_text = []
  if isinstance(data, dict):
    if "text" in data and isinstance(data["text"], str):
      extracted_text.extend(chunk_text_from_directory(data["text"], chunk_size=512))
  elif isinstance(data, list):
    for item in data:
      extracted_text.extend(extract_text(item))
  return extracted_text

def chunk_text_from_directory(text, chunk_size):
  all_texts = [text]
  chunks = []
  for i in range(0, len(all_texts), chunk_size):
    chunks.append(all_texts[i : i + chunk_size])
  return chunks

def load_documents(data_folder):
  documents = []
  for filename in os.listdir(data_folder):
    if filename.endswith(".json"):
      with open(os.path.join(data_folder, filename), "r", encoding="utf-8") as file:
                data = json.load(file)
                chunks = extract_text(data)
                documents.extend(chunks)
  return documents
