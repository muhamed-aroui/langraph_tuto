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

class DiagnosisSchema(BaseModel):
    issue_type : Annotated[Literal["UX","Performance","Bug","Support","Other"], Field(description= "Category of issue that is mentioned in the negative comment")]
    tone: Annotated[Literal["Angry","Frustrated","Calm","Sarcastic","Disapointed"],Field(description="Emotional tone present in the review")]
    urgency : Annotated[Literal["Hight","Midium","Low"],Field(description="Urgency of to fix the issue")]

structured_model=  model.with_structured_output(SentimentSchema)
structured_model2 = model.with_structured_output(DiagnosisSchema)


class ReviewState(TypedDict):
    review :str
    sentiment : Literal["Positive","Negative"]
    diagnosis : dict
    response :str

def find_sentiment(state :ReviewState):
    review = state['review']
    prompt = f"Give the sentiment of this review \n{review}"
    result= structured_model.invoke(prompt)
    return {"sentiment":result.sentiment}

def check_sentiment(state:ReviewState):
    if state["sentiment"] == "Positive":
        return "positive_response"
    else:
        return "run_diagnosis"
    
def positive_response(state:ReviewState):
    review = state['review']
    prompt = f"Write a thank you message to respond this review \n{review}"
    result = model.invoke(prompt)
    return {"response" : result.content}

def run_diagnosis(state:ReviewState):
    review = state['review']
    prompt = f"Dignose this negative review, return the issue type, tone and urgency:\n{review}"
    result = structured_model2.invoke(prompt)
    return {"diagnosis":result.model_dump()}

def negative_response(state:ReviewState):
    diagnosis = state['diagnosis']
    prompt = f"You are a support assistant.\nThe user had an this type of issue {diagnosis['issue_type']} sounded {diagnosis['tone']},marked the urgency as {diagnosis['urgency']}\nYou have to write an empathic and helpful resolution response"
    result = model.invoke(prompt)
    return {"response" : result.content}

graph = StateGraph(ReviewState)
graph.add_node("find_sentiment",find_sentiment)
graph.add_node("positive_response",positive_response)
graph.add_node("run_diagnosis",run_diagnosis)
graph.add_node("negative_response",negative_response)

graph.add_edge(START,"find_sentiment")
graph.add_conditional_edges("find_sentiment",check_sentiment)
graph.add_edge("run_diagnosis","negative_response")
graph.add_edge("positive_response",END)
graph.add_edge("negative_response",END)

workflow  = graph.compile()
initial_state = {"review" :"I Added my credit card on website but it charged me some unreasonable fees , I WANT MY REFUND RIGHT NOW "}
final_state = workflow.invoke(initial_state)
print(final_state)