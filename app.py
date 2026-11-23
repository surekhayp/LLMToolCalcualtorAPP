from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage,ToolMessage
from tools import add, divide, multiply, subtract


MODEL="llama3.2"
# Group tools into a list and a lookup dictionary
tools = [add, subtract, multiply, divide]
tool_map = {t.name: t for t in tools}
print(f"Available tools: {list(tool_map.keys())}")
# print(f"Tool map: {tool_map.values()}")
# print(f"Tool map: {tool_map}")

# 2. Initialize ChatOllama and bind the tools
llm = ChatOllama(model=MODEL, temperature=0)
llm_with_tools = llm.bind_tools(tools)

# 3. Create the execution flow
def run_calculator_query(user_query: str):
    print(f"User: {user_query}")
    messages = [HumanMessage(content=user_query)]

    # Step A: Ask the LLM (it will decide if a tool is needed)
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

    # Step B: Check if the LLM requested any tool calls
    if ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"  -> Calling tool '{tool_name}' with args: {tool_args}")

            # Execute the matching Python function
            selected_tool = tool_map[tool_name]
            tool_output = selected_tool.invoke(tool_args)

            # Append the tool's output back into the conversation history
            messages.append(
                ToolMessage(
                    content=str(tool_output), tool_call_id=tool_call["id"]
                )
            )
        # Step C: Send the tool output back to the LLM for the final natural response
        final_response = llm_with_tools.invoke(messages)
        print(f"Assistant: {final_response.content}\n")
    else:
        print(f"Assistant: {ai_msg.content}\n")


# Test the calculator tools
if __name__ == "__main__":
    run_calculator_query("What is 145.5 minus 32.2?")
    run_calculator_query("Can you divide 144 by 12?")
    run_calculator_query("Find the answer of 125 * 5")
    run_calculator_query("What happens if I divide 50 by 0?")

