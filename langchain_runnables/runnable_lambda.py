from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)

load_dotenv()

# Prompt
prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

# LLM
llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=200
    )
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

# Step 1 — Generate joke
joke_gen_chain = RunnableSequence(prompt1, model, parser)

# Step 2 — Run in parallel: joke + word count
parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),  # fix trailing space AND passthrough
    "word_count": RunnableLambda(lambda x: len(x.split()))  # call split()
})

# Step 3 — Sequence: joke generation → parallel processing
final_runnable_chain = RunnableSequence(joke_gen_chain, parallel_chain)

# Invoke
result = final_runnable_chain.invoke({"topic": "women"})

final_result = f"{result['joke']}\nWord count: {result['word_count']}"
print(final_result)
