from dotenv import load_dotenv
from groq import Groq
import os
import streamlit as st
import time
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import streamlit as st
import requests

load_dotenv()
embedder = SentenceTransformer("BAAI/bge-base-en", device = 'cpu') #cpu because streamlit cloud doesnt have cpu in free tier.
groq = Groq(api_key = os.getenv('GROQ_API'))

git_link = "https://raw.githubusercontent.com/Ghnkrk/chatbot/refs/heads/main/data/kct_enriched_data.json"
response = requests.get(git_link)
data = response.json()
content = [entry['content'] for entry in data]

def load_index():
    index = faiss.read_index('kct_index.faiss')
    with open("kct_metadata.pkl",'rb') as f:
        metadata = pickle.load(f)
    return index, metadata

def get_embed(query):
     return embedder.encode(query)
     

def search_db(content, metadata, index, query):
        D, I = index.search(np.array([get_embed(query)]), k=1)

        top_content = content[I[0][0]]
        top_metadata = metadata[I[0][0]]

        return top_content, top_metadata

def infer_model(history, top_content, top_metadata):
     messages = [
          {
               "role" : "system",
               "content" : f"You are an assistant that answers user queries regarding the college information. The context is provided here:{top_content}. Also provide the relevant url source to the user from the metadata here: {top_metadata}. Be friendly yet accurate about your answers."
            }  
     ]
     messages.extend(history)
     completion = groq.chat.completions.create(
          messages=messages,
          model='llama3-8b-8192'
     )
     response = completion.choices[0].message.content
     return response

def response_generator(response):
        for c in response:
                yield c
                time.sleep(0.007)

index, metadata = load_index()


st.title("CHATBOT")

if st.session_state.get("messages") is None:
        st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Enter your query:")
if query:
        with st.chat_message("user"):
                st.markdown(query)
        st.session_state.messages.append({
                "role" : "user",
                "content" : f'{query}'
        }) 
        MAX_TURNS = 10
        if len(st.session_state.messages) > MAX_TURNS:
            st.session_state.messages = st.session_state.messages[-MAX_TURNS:]
        top_content, top_metadata = search_db(content= content, metadata= metadata, index= index, query= query)
        response = infer_model(history= st.session_state.messages, top_content=top_content, top_metadata=top_metadata)

        with st.chat_message("system"):
                st.write_stream(response_generator(response))
        st.session_state.messages.append({
                "role":"system",
                "content" : f'{response}'
        })
