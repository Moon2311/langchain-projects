import os
import re
import time
import requests
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# -------------------------------
# CREATE THE BASE LLM
# -------------------------------
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.3-70B-Instruct",
    max_new_tokens=1024,
    temperature=0.2,
    huggingfacehub_api_token=hf_token,
)

chat_llm = ChatHuggingFace(llm=llm)

# -------------------------------
# TOOLS
# -------------------------------
@tool
def get_conversion_factor(base_currency: str, target_currency: str):
    """Get conversion rate between two currencies."""
    url = f"https://v6.exchangerate-api.com/v6/c754eab14ffab33112e380ca/pair/{base_currency}/{target_currency}"
    response = requests.get(url)
    data = response.json()
    conversion_rate = data["conversion_rate"]
    print("Conversion Rate Data:", conversion_rate)
    return str(conversion_rate)


@tool
def convert(base_currency_value: float, conversion_rate: float) -> float:
    """Convert amount based on conversion rate."""
    return base_currency_value * conversion_rate


# -------------------------------
# BIND TOOLS TO THE MODEL
# -------------------------------
llm_with_tools = chat_llm.bind_tools([get_conversion_factor, convert])

messages = [
    HumanMessage(
        "What is the conversion factor between PKR and USD, and convert 10 USD to PKR?"
    )
]

# -------------------------------
# SAFE INVOKE WITH RETRIES
# -------------------------------
def safe_invoke(model, messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return model.invoke(messages)
        except Exception as e:
            err = str(e)
            if "429" in err or "rate limit" in err.lower():
                print("\n⚠️ Rate limit hit!")
                match = re.search(r"retry in (\d+)", err)
                wait_time = int(match.group(1)) if match else 30
                print(f"⏳ Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise e
                time.sleep(5)
    raise Exception("Failed after max retries!")

# -------------------------------
# 1) ASK MODEL + HANDLE TOOLS
# -------------------------------
print("--- Sending initial request to model ---")
ai_message = safe_invoke(llm_with_tools, messages)
messages.append(ai_message)

print("\n--- AI Message ---")
print(f"Content: {ai_message.content}")
print(f"Tool Calls: {getattr(ai_message, 'tool_calls', [])}")

# -------------------------------
# 2) EXECUTE ALL TOOL CALLS WITH REAL RESULTS
# -------------------------------
tool_results = {}
if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
    for call in ai_message.tool_calls:
        tool_name = call.get("name")
        tool_args = call.get("args", {})
        tool_id = call.get("id")

        print(f"\n🔧 Executing tool: {tool_name}")
        print(f"   Args: {tool_args}")

        if tool_name == "get_conversion_factor":
            result = get_conversion_factor.invoke(tool_args)
            # Save actual conversion rate for later
            tool_results["conversion_rate"] = float(result)
            messages.append(ToolMessage(content=result, tool_call_id=tool_id))

        elif tool_name == "convert":
            # Use real conversion rate from get_conversion_factor
            tool_args["conversion_rate"] = tool_results.get("conversion_rate", tool_args["conversion_rate"])
            result = convert.invoke(tool_args)
            messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))

        else:
            result = f"Unknown tool: {tool_name}"
            messages.append(ToolMessage(content=result, tool_call_id=tool_id))

# -------------------------------
# 3) SEND TOOL RESULTS BACK TO MODEL
# -------------------------------
print("\n--- Sending Tool Results Back to Model ---")
final_response = safe_invoke(llm_with_tools, messages)

print("\n--- Final Output ---")
print(final_response.content)
