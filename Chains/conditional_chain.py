from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

# -------------------
# Model
# -------------------
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0,
        max_new_tokens=1,
    )
)
model = ChatHuggingFace(llm=llm)

# -------------------
# Parser for string output
# -------------------
parser = StrOutputParser()

# -------------------
# Pydantic wrapper for branch access
# -------------------
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative']
import re
def extract_sentiment(text: str) -> Feedback:
    # Find 'positive' or 'negative' anywhere in the text
    match = re.search(r'\b(positive|negative)\b', str(text).lower())
    if match:
        return Feedback(sentiment=match.group(1))
    else:
        # fallback if model output is unexpected
        return Feedback(sentiment='negative')


wrap_feedback = RunnableLambda(extract_sentiment)

# -------------------
# Prompts
# -------------------
prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negative:\n{feedback}',
    input_variables=['feedback']
)

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback:\n{feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback:\n{feedback}',
    input_variables=['feedback']
)

# -------------------
# Chains
# -------------------
# Step 1: classify sentiment as string
classifier_chain = prompt1 | model | parser

# Step 2: wrap into Feedback object for branch
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "Could not determine sentiment")
)

# Step 3: Full chain
chain = classifier_chain | wrap_feedback | branch_chain

# -------------------
# Run
# -------------------
result = chain.invoke({'feedback': 'This is a beautiful phone'})
print(result)

# -------------------
# Optional: Graph visualization
# -------------------
try:
    chain.get_graph().print_ascii()
except AttributeError:
    print("Graph visualization requires langchain-core >= 0.2.10")
