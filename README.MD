# PF2e-RAG: Pathfinder 2e RAG Assistant

A lightweight retrieval-augmented generation (RAG) system using SentenceTransformers, FAISS, and LLaMA 3 to answer Pathfinder 2e questions!



# How to Customize Data

- The folder 'archives-of-nethys-scraper/parsed/' is where the data is taken from Parsed JSON files
- You can delete these files and add your own data if you wish to customize
- After changing data, **delete the what is in the 'index' folder** to force the system to rebuild a fresh index
- If a .pk1 file and index already exist in 'index/', the system loads the saved index to save time

# User Instructions

1. When prompted, type in your question about Pathfinder 2e. Ex: How much damage does fireball do?
2. The program will search and generate an answer based on the data available.
3. When you are done, type "exit" to end the session

Enjoy the Archives!

# Setup Instructions

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

# Example Test Runs

**Question:** How do troop creatures work?

**Answer:** Troop creatures are a type of creature that has an organization that is organized as a group of creatures. In the case of the troop creature, the group of creatures are small in size and are usually organized into two main categories; infantry and cavalry. The infantry is made up of soldiers or creatures that move on foot, while the cavalry is made up of mounted combat units. The troop creature has a strength and a weakness, which is the type of damage it is most susceptible to. The troop creature has a weakness to area damage, meaning that it is most susceptible to damage from fire and other sources of area damage.

**Question:** What triggers Attack of Opportunity?

**Answer:** Attack of Opportunity is triggered when a creature within your reach uses a concentrate action, in addition to manipulate and move actions. Furthermore, you disrupt a triggering concentrate or manipulate action if your critically hits.

