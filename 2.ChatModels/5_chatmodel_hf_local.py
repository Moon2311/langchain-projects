from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
import os
 
import os

# set HF cache to a folder you own
os.environ["HF_HOME"] = "/home/talha/hf_cache"  # create this folder first
os.makedirs("/home/talha/hf_cache", exist_ok=True)


llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=200
    )
    

)
model= ChatHuggingFace(llm=llm)
result = model.invoke("What is the capital of panjab pakistan")

print(result.content)