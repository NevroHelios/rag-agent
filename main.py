import os

from colorama import Fore, Style, init
from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from hybrid_search import load_or_create_vectore_store


init(autoreset=True) 
load_dotenv()


llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.1,  # good for rag :reddit
    groq_api_key=os.getenv("GROQ_API_KEY"),
)


@tool
def query_vector_store(query: str) -> str:
    """
    Queries a vector store for specific university policies or course information like final exam criteria.
    Use this tool ONCE at the beginning of the conversation to gather context based on the user's initial query.
    Args:
        query (str): The specific question or topic to search for (e.g., 'final exam criteria', 'grading policy').
    Returns:
        str: Relevant document excerpts separated by newlines, or a 'No relevant documents found' message.
    """
    print(
        f"{Fore.CYAN}{Style.BRIGHT}--- Calling Tool: query_vector_store ---{Style.RESET_ALL}"
    )
    print(f"{Fore.CYAN}Query: {query}{Style.RESET_ALL}")
    try:

        store = load_or_create_vectore_store()
        retriever = store.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)
        if not docs:
            result = "No relevant documents found in the vector store for this query."
        else:
            result = f"Retrieved documents for query '{query}':\n\n" + "\n\n".join(
                doc.page_content for doc in docs
            )

        print(f"{Fore.CYAN}Result:\n{result}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{Style.BRIGHT}--- Tool Finished ---{Style.RESET_ALL}")
        return result
    except Exception as e:
        print(f"{Fore.RED}Error during vector store query: {e}{Style.RESET_ALL}")
        print(Fore.RED + "Maybe the Qdrant server is down or unreachable." + Style.RESET_ALL)
        return f"Error querying vector store: {e}"


# --- Agent Setup ---
tools = [query_vector_store]
llm_with_tools = llm.bind_tools(tools)


# Node functions now accept the standard MessagesState type hint
def call_model(state: MessagesState):
    """Invokes the LLM with the current message history."""
    print(f"{Fore.BLUE}{Style.BRIGHT}--- Calling Model ---{Style.RESET_ALL}")

    messages = state["messages"]
    print(f"{Fore.BLUE}Messages sent to LLM ({len(messages)}):{Style.RESET_ALL}")
    
    ## debuggin
    for i, msg in enumerate(messages):
        prefix = f"{Fore.BLUE}  {i + 1}. Type: {type(msg).__name__}, "
        if isinstance(msg, SystemMessage):
            print(f"{prefix}Content:\n{msg.content[:500]}...{Style.RESET_ALL}")
        else:
            print(f"{prefix}Content: {msg.content}{Style.RESET_ALL}")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"{Fore.BLUE}     Tool Calls: {msg.tool_calls}{Style.RESET_ALL}")
        if isinstance(msg, ToolMessage):
            print(
                f"{Fore.GREEN}     Tool ID: {msg.tool_call_id}{Style.RESET_ALL}"
            )

    response = llm_with_tools.invoke(messages)
    print(f"{Fore.BLUE}Model Response Raw:{Style.RESET_ALL}")
    print(
        f"{Fore.BLUE}  Type: {type(response).__name__}, Content: {response.content}{Style.RESET_ALL}"
    )
    if response.tool_calls:
        print(f"{Fore.BLUE}  Tool Calls: {response.tool_calls}{Style.RESET_ALL}")

    # Return the new message(s) to be appended by MessagesState
    return {"messages": [response]}


tool_node = ToolNode(tools)



# Conditional edge function now accepts MessagesState
def should_continue(state: MessagesState) -> str:
    """Decides whether to call tools or end the conversation."""
    print(
        f"{Fore.MAGENTA}{Style.BRIGHT}--- Checking Condition 'should_continue' ---{Style.RESET_ALL}"
    )
    last_message = state["messages"][-1]
    print(
        f"{Fore.MAGENTA}Last message type: {type(last_message).__name__}{Style.RESET_ALL}"
    )

    if isinstance(last_message, AIMessage) and getattr(
        last_message, "tool_calls", None
    ):
        print(
            f"{Fore.MAGENTA}Decision: Call Tools (AIMessage has tool_calls){Style.RESET_ALL}"
        )
        return "call_tool"
    else:
        print(
            f"{Fore.MAGENTA}Decision: End (AIMessage has no tool_calls, or last message is not AIMessage){Style.RESET_ALL}"
        )
        return END

# --- workflow ---
workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("call_tool", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent", should_continue, {"call_tool": "call_tool", END: END}
)
workflow.add_edge("call_tool", "agent")
app = workflow.compile()


def run_agent(query: str):
    print(f"\n{Fore.GREEN}{Style.BRIGHT}--- Running Agent ---{Style.RESET_ALL}")
    print(f"{Fore.GREEN}User Query: {query}{Style.RESET_ALL}")

    system_message_content = (
        "You are a helpful assistant answering questions about university policies using a vector store. "
        "Your process MUST be as follows:\n"
        "1. Receive the user's query.\n"
        "2. **Immediately** call the `query_vector_store` tool ONE TIME to retrieve relevant policy documents based on the user's query. Construct the tool's 'query' argument appropriately.\n"
        "3. Wait for the tool results (which will appear as a `ToolMessage` in the history).\n"
        "4. **After receiving the `ToolMessage` results**, analyze the ENTIRE message history (including your previous tool call and the results). Synthesize the information from the retrieved documents into a clear, concise answer to the user's original question. \n"
        "5. **DO NOT** call `query_vector_store` again after you have received the results for the initial query. Only generate the final answer based on the provided tool results.\n"
        "6. If the tool returns 'No relevant documents found', inform the user you couldn't find specific information in the policies based on their query."
    )
    system_message = SystemMessage(content=system_message_content)

    
    initial_state = {"messages": [system_message, HumanMessage(content=query)]}

    # Increase recursion limit slightly just in case
    final_state = app.invoke(initial_state, {"recursion_limit": 10})

    # Extract final answer 
    final_answer = "Agent did not produce a final answer." 
    if final_state and "messages" in final_state:
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                final_answer = msg.content
                break
        if final_answer == "Agent did not produce a final answer." and isinstance(
            final_state["messages"][-1], AIMessage
        ):
            final_answer = final_state["messages"][-1].content
        elif final_answer == "Agent did not produce a final answer.":
            print(
                f"{Fore.YELLOW}Could not extract final AIMessage answer. Last message was: {final_state['messages'][-1]}{Style.RESET_ALL}"
            )
            final_answer = "Sorry, I encountered an issue processing the final response. Please check the logs."

    print(f"\n{Fore.YELLOW}{Style.BRIGHT}--- Final Answer ---{Style.RESET_ALL}")
    print(Fore.YELLOW + final_answer + Style.RESET_ALL)

    # debuggin
    print(f"\n{Fore.WHITE}{Style.BRIGHT}--- Full Message History ---{Style.RESET_ALL}")
    if final_state and "messages" in final_state:
        for i, msg in enumerate(final_state["messages"]):
            prefix = f"{Fore.WHITE}{i + 1}. {type(msg).__name__}: "
            if isinstance(msg, SystemMessage):
                print(f"{prefix}\n{msg.content[:200]}...{Style.RESET_ALL}")
            else:
                print(f"{prefix}{msg.content}{Style.RESET_ALL}")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                print(f"{Fore.WHITE}   Tool Calls: {msg.tool_calls}{Style.RESET_ALL}")
            if isinstance(msg, ToolMessage):
                print(f"{Fore.GREEN}   Tool ID: {msg.tool_call_id}{Style.RESET_ALL}")

    return final_answer


# --- Main Execution ---
if __name__ == "__main__":
    user_query = "I got low marks on mad1 so should I move to mad2 or repeat?"
    run_agent(user_query)
