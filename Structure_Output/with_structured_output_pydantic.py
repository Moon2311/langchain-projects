import json
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Load environment variables
load_dotenv()

# Load HuggingFace endpoint LLM
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=1024,
)

model = ChatHuggingFace(llm=llm)

# ---------------------------------------------------------
# Pydantic Schema
# ---------------------------------------------------------
class Review(BaseModel):
    key_themes: list[str] = Field(description="Key themes discussed in the review")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(description="Sentiment of the review")
    pros: Optional[list[str]] = Field(default=None, description="Pros of the product")
    cons: Optional[list[str]] = Field(default=None, description="Cons of the product")
    name: Optional[str] = Field(default=None, description="Name of the reviewer")

# JSON parser using the Pydantic model
parser = JsonOutputParser(pydantic_object=Review)

# Prompt template with format instructions
prompt = PromptTemplate(
    template="""
Extract structured information from the following review.
Return ONLY valid JSON. No explanation.

Review text:
{text}

{format_instructions}
""",
    input_variables=["text"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

# Combine prompt + model + parser
chain = prompt | model | parser

# ---------------------------------------------------------
# Input review
# ---------------------------------------------------------
review_text = """
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! 
The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, 
or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast 
charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. 
What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images 
even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x 
loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still 
comes with bloatware—why do I need five different Samsung apps for things Google already provides? 
The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Review by Nitish Singh
"""

# ---------------------------------------------------------
# Run the chain
# ---------------------------------------------------------
result = chain.invoke({"text": review_text})
print(result)
review= Review(**result)

print(review.name)
print(review.summary)
print(review.key_themes)

# Print final structured result
# print(result)
