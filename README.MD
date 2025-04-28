# PF2e-RAG: Pathfinder 2e RAG Assistant

A lightweight retrieval-augmented generation (RAG) system using SentenceTransformers, FAISS, and LLaMA 3 to answer Pathfinder 2e questions!

---

# How to Customize Data

---

- The folder 'archives-of-nethys-scraper/parsed/' is where the data is taken from Parsed JSON files
- You can delete these files and add your own data if you wish to customize
- After changing data, **delete the what is in the 'index' folder** to force the system to rebuild a fresh index
- If a .pk1 file and index already exist in 'index/', the system loads the saved index to save time

---

# User Instructions

---
1. When prompted, type in your question about Pathfinder 2e. Ex: How much damage does fireball do?
2. The program will search and generate an answer based on the data available.
3. When you are done, type "exit" to end the session

Enjoy the Archives!

---

# Setup Instructions

---
1. Clone the Repository

2. Activate a virtual enviroment:
EX: python -m venv .venv
.venv/scripts/activate

3. Install required packages:
pip install -r requirements.txt

4. Gain Access to Model:
PF2e RAG uses a LLaMA 3 model that you can request access to on Hugging Face. 
Go to https://huggingface.co/meta-llama/Llama-3.2-1B and request access. You should be approved within
the day

5. Set your Hugging Face Token:
$env:HF_TOKEN="your_token_here"

You are then all set to run "python main.py" to run PF2e-RAG!

