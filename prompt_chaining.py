from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

model_name = 'gemini-2.5-flash'
model  = ChatGoogleGenerativeAI(model=model_name)

class BlogState(TypedDict):
    title : str
    outline : str
    content :str

def create_outline(state:BlogState)->BlogState:
    title = state['title']
    prompt = f'write a detailed outiline of a blog for this topic :{title}'

    state['outline'] = model.invoke(prompt).content
    return state

def create_blog(state:BlogState)->BlogState:
    outline = state['outline']
    title =state['title']
    prompt =f'write a blog on the topic of {title} following this outilne {outline}'
    state['content']= model.invoke(prompt).content
    return state

graph = StateGraph(BlogState)

graph.add_node('create_outline',create_outline)
graph.add_node('create_blog',create_blog)

graph.add_edge(START,'create_outline')
graph.add_edge('create_outline','create_blog')
graph.add_edge('create_blog',END)

workflow = graph.compile()
initial_state= {'title':'The bubble of AI'}
final_state =workflow.invoke(initial_state)
print(final_state)