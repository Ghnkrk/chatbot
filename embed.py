import json
from sentence_transformers import SentenceTransformer
import requests
import faiss
import numpy as np
import pickle

def embed_and_store():
        git_link = "https://raw.githubusercontent.com/Ghnkrk/chatbot/refs/heads/main/data/kct_enriched_data.json"
        response = requests.get(git_link)
        data = response.json()

        embedder = SentenceTransformer("BAAI/bge-base-en")

        embedding = []
        metadata = []
        content = []
        for entry in data:
            text = entry['content']
            meta = {
                'url' : entry.get('url'),
                'section' : entry.get('section')
            }
            vectors = embedder.encode(content, show_progress_bar=True)
            embedding.append(vectors)
            metadata.append(meta)
            content.append(text)
        return content, embedding, metadata 

def vector_store(embedding):
        embed_array = np.vstack(embedding)
        embed_dim = embed_array.shape[1]

        index = faiss.IndexFlatL2(embed_dim)
        index.add(embed_array)

        return index

def process():
       content, embedding, metadata = embed_and_store()
       index = vector_store(embedding= embedding)

       faiss.write_index(index, "kct_index.faiss")
       with open("kct_metadata.pkl", 'wb') as f:
              pickle.dump(metadata, f)

process()