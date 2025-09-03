from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from pydantic import BaseModel,Field
import operator

load_dotenv()

model_name = 'gemini-2.5-flash'
model  = ChatGoogleGenerativeAI(model=model_name)

class EvaluationSchema(BaseModel):
    feedback: Annotated[str,Field(description= "A detailed feedack of the essay")]
    score: Annotated[int,Field(description="score of the essay out of 10", ge=0,le=10)]

structured_model = model.with_structured_output(EvaluationSchema)

essay ="""The current surge in artificial intelligence feels like a growing bubble.
Investment and hype have skyrocketed, with companies racing to adopt AI tools.
While the technology has clear benefits, its limitations are often overlooked.
Many startups promise groundbreaking innovation without sustainable business models.
This creates inflated expectations that may not match reality.
If the bubble bursts, weaker players will disappear, but core progress will remain.
Ultimately, the AI bubble may correct itself, leaving behind lasting advancements."""

class EssayState(TypedDict):
    essay : str
    fdb_language_quality : str
    fdb_topic_analysis : str
    fdb_clarity_of_thought : str
    fdb_overall : str
    indivudual_score : Annotated[list[int],operator.add]
    avg_score : float


def eval_language_quality(state: EssayState):
    prompt= f"Evaluate the language quality of this essay, provide a feedback and give a score out of 10:\n{state['essay']}"
    result = structured_model.invoke(prompt)
    return {'fdb_language_quality':result.feedback,'indivudual_score':[result.score]}

def eval_topic_analysis(state: EssayState):
    prompt= f"Evaluate the depth of analysis of this essay, provide a feedback and give a score out of 10:\n{state['essay']}"
    result = structured_model.invoke(prompt)
    return {'fdb_topic_analysis':result.feedback,'indivudual_score':[result.score]}

def eval_clarity_of_thought(state: EssayState):
    prompt= f"Evaluate the clarity of thought of this essay, provide a feedback and give a score out of 10:\n{state['essay']}"
    result = structured_model.invoke(prompt)
    return {'fdb_clarity_of_thought':result.feedback,'indivudual_score':[result.score]}


def eval_final(state: EssayState):
    prompt = f"Based on the following feedbacks create a summarized feedback:\nLanguage quality - {state['fdb_language_quality']}\nDepth of analysis - {state['fdb_topic_analysis']}\nClarity of thought - {state['fdb_clarity_of_thought']}"
    fdb_overall  = model.invoke(prompt).content
    avg_score = round(sum(state['indivudual_score'])/len(state['indivudual_score']),2)
    return {'fdb_overall':fdb_overall, 'avg_score':avg_score}

graph = StateGraph(EssayState)
graph.add_node('eval_language_quality',eval_language_quality)
graph.add_node('eval_topic_analysis',eval_topic_analysis)
graph.add_node('eval_clarity_of_thought',eval_clarity_of_thought)
graph.add_node('eval_final',eval_final)


graph.add_edge(START,'eval_language_quality')
graph.add_edge(START,'eval_topic_analysis')
graph.add_edge(START,'eval_clarity_of_thought')

graph.add_edge('eval_language_quality','eval_final')
graph.add_edge('eval_topic_analysis','eval_final')
graph.add_edge('eval_clarity_of_thought','eval_final')

graph.add_edge('eval_final',END)

workflow= graph.compile()


initial_state = {'essay':essay}

final_state = workflow.invoke(initial_state)

print(final_state)