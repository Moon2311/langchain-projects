from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
import os
from dotenv import load_dotenv

load_dotenv()

# --- 1. Tool definition ---
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

# --- 2. Gemini LLM setup ---
# Make sure your API key is loaded
if not os.getenv("GOOGLE_API_KEY"):
    print("Error: GOOGLE_API_KEY not found in environment variables.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    # If the key isn't in env, paste it here temporarily to test: google_api_key="...",
    max_retries=6,
    request_timeout=60
)

# --- 3. Bind tools ---
# This line fails if the library is outdated
try:
    llm_with_tools = llm.bind_tools([multiply])
except AttributeError:
    print("CRITICAL ERROR: Your 'langchain-google-genai' version is too old. Please run: pip install -U langchain-google-genai")
    exit()

# --- 4. Test ---
print("Sending request to Gemini...")

# response = llm_with_tools.invoke("Multiply 7 and 9")


from langchain_core.messages import HumanMessage, ToolMessage

# ... (Your existing code: Tool definition, LLM setup, Bind tools) ...

# --- 4. First Step: User Query (LLM decides to call a tool) ---
user_query = "Multiply 7 and 9"
print(f"Sending initial request: '{user_query}'")
tool_call_response = llm_with_tools.invoke(user_query)

print("\n--- Step 1 Output (LLM Request) ---")
print(f"LLM decided to call: {tool_call_response.tool_calls}")

# --- 5. Second Step: Execute Tool ---
tool_messages = []
final_result = None # To store the final calculation

if tool_call_response.tool_calls:
    for tool_call in tool_call_response.tool_calls:
        # 1. Extract arguments
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"] # Important for the response message

        # 2. Execute the function (in a real app, you'd look up the function)
        if tool_name == "multiply":
            final_result = multiply.invoke(tool_args)

            print(f"Executed multiply({tool_args['a']}, {tool_args['b']}). Result: {final_result}")

            # 3. Create the ToolMessage object with the result
            tool_messages.append(
                ToolMessage(
                    content=str(final_result),
                    tool_call_id=tool_id,
                )
            )

# --- 6. Third Step: Send result back to LLM for final answer ---
if tool_messages:
    # Build the full conversation history: 
    # [User Query, Tool Call Request, Tool Execution Result]
    messages = [
        HumanMessage(content=user_query),           # Original user message
        tool_call_response,                         # LLM's request for the tool
        *tool_messages                              # The result of the tool execution
    ]
    
    print("\n--- Step 3 Input (Tool Result Sent Back) ---")
    print("Sending ToolMessage back to LLM...")

    # Invoke the LLM again with the complete conversation history
    final_response = llm_with_tools.invoke(messages)

    print("\n--- Final Output (LLM Answer) ---")
    print(final_response.content)
    # Expected output: "The result of multiplying 7 and 9 is 63."