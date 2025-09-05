from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph,START,END
from typing import Annotated,Literal,TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


model_name = 'gemini-2.5-flash'
model  = ChatGoogleGenerativeAI(model=model_name)


class SentimentSchema(BaseModel):
    sentiment : Annotated[Literal["Positive","Negative"],Field(description="Sentiment of the review")]

structured_model=  model.with_structured_output(SentimentSchema)
# prompt = "give the sentiment of this review -  the usage of the product is hard "
# result = structured_model.invoke(prompt)


class ReviewState(TypedDict):
    review :str
    sentiment : Literal["Positive","Negative"]
    diagnosis : dict
    response :str

def find_sentiment(state :ReviewState):
    review = state["review"]
    prompt = f"Give the sentiment of this review \n{review}"
    result= structured_model.invoke(prompt)
    return {"sentiment":result.sentiment}


graph = StateGraph(ReviewState)
graph.add_node("find_sentiment",find_sentiment)

graph.add_edge(START,"find_sentiment")
graph.add_edge("find_sentiment",END)

workflow  = graph.compile()
initial_state = {"review" :"I tried to search but it keeps bugging "}
final_state = workflow.invoke(initial_state)
