from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
    RunnableBranch
)

load_dotenv()

# Prompts
prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text:\n{text}',
    input_variables=['text']
)

# LLM
llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=300
    )
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

# Main report generation chain
report_gen_chain = prompt1 | model | parser

# Branch: summarize only if text length > 150 words
branch_chain = RunnableBranch(
    (
        lambda text: len(text.split()) > 150,
        RunnableLambda(lambda t: {"text": t}) | prompt2 | model | parser
    ),
    RunnablePassthrough()
)

# Final chain: generate → then summarize (if needed)
final_chain = RunnableSequence(report_gen_chain, branch_chain)

print(final_chain.invoke({'topic': 'Russia vs Ukraine'}))
