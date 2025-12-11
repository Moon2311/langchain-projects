import os
import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate,load_prompt
# -----------------------------------------------------
# 1) Fix HuggingFace Cache Directory Permission Problems
# -----------------------------------------------------
os.environ["HF_HOME"] = "/home/talha/hf_cache"
os.makedirs("/home/talha/hf_cache", exist_ok=True)

# -----------------------------------------------------
# 2) Load Local HF Model Using Pipeline
# -----------------------------------------------------
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=200,
        device_map="auto",           # automatically manage CPU memory
        torch_dtype="float32",
        low_cpu_mem_usage=True       # prevents meta tensor error
    )
)


model = ChatHuggingFace(llm=llm)

# -----------------------------------------------------
# 3) Streamlit App
# -----------------------------------------------------
st.title("Research Tool - TinyLlama")



paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )
 
template = load_prompt('template.json')



if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input
    })
    st.write(result.content)
