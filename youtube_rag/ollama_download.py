# app.py — just add/replace this part
from langchain_community.llms import Ollama
import streamlit as st

@st.cache_resource
def load_llm():
    """Loads Phi-3 locally via Ollama — offline, fast, unlimited"""
    return Ollama(
        model="phi3",           # or "phi3:mini-128k" for very long videos
        temperature=0.3,
        num_predict=2048        # long answers = no cutoff!
    )

# Example usage in your generate_answer function
llm = load_llm()

# For summary
summary = llm.invoke("Provide a detailed summary of the entire video bad on the transcript context provided earlier...")
print(summary)

# For Q&A
answer = llm.invoke(f"Context: {context}\n\nQuestion: {user_question}\nAnswer:")
print(answer)