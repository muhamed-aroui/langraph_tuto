from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

model_name = 'gemini-2.5-flash'
model  = ChatGoogleGenerativeAI(model=model_name)

class StateLLM(TypedDict):
    question:str
    answer:str

graph = StateGraph(StateLLM)

def llm_qa(state:StateGraph)->StateGraph:
    q = state['question']
    
    prompt = f'answer the following question: {q}'

    a = model.invoke(prompt).content
    
    state['answer'] = a
    return state

graph.add_node('llm_qa',llm_qa)

graph.add_edge(START,'llm_qa')
graph.add_edge('llm_qa',END)

workflow  = graph.compile()

initial_state= {'question':'what is the best place to visit in USA'}

final_state = workflow.invoke(initial_state)
print(final_state['answer'])


