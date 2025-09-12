from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage
from typing import Annotated,TypedDict
from dotenv import load_dotenv

load_dotenv()

model_name = 'gemini-2.5-flash'
model  = ChatGoogleGenerativeAI(model=model_name)

class ChatState(TypedDict):

    messages : Annotated[list[BaseMessage],add_messages]

def chat_node(state: ChatState):
    prompt = state["messages"]
    result = model.invoke(prompt)
    return {"messages":[result]}


graph = StateGraph(ChatState)

graph.add_node("chat_node",chat_node)

graph.add_edge(START,"chat_node")
graph.add_edge("chat_node",END)

checkpoint = MemorySaver()
chatbot = graph.compile(checkpointer=checkpoint)


# initial_state = {"messages": [HumanMessage("what is the age of ammi tabboun")]}

# response = chatbot.invoke(initial_state)["messages"][-1].content
thread_id = "1"
while True:
    
    user_message = input("Type here: ")

    if user_message.strip().lower() in ['exit','bye','quit']:
        break
    
    print("User: ",user_message)

    config = {'configurable':{'thread_id':thread_id}}

    response = chatbot.invoke({"messages":[HumanMessage(user_message)]},config=config)

    print("AI: ", response["messages"][-1].content)





