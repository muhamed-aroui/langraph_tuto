from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage 
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv
from typing import TypedDict, Annotated,Literal
from pydantic import BaseModel,Field

load_dotenv()


model_name = 'gemini-2.5-flash'

generator_llm = ChatGoogleGenerativeAI(model=model_name)
evaluator_llm = ChatGoogleGenerativeAI(model=model_name)
optimazer_llm = ChatGoogleGenerativeAI(model=model_name)

class EvaluatorOut(BaseModel):
    evaluation: Annotated[Literal["approved","needs_improvment"],Field(description= "approve or not the tweet")]
    feedback: Annotated[str,Field(description="One paragraph explaining the strengths and weaknesses")]

structured_evaluator = evaluator_llm.with_structured_output(EvaluatorOut)

class tweetState(TypedDict):
    topic : str
    tweet : str
    evaluation : Literal["approved", "needs_improvment"]
    feedback : str
    iteration : int
    max_iteration :int


def gen_tweet(state : tweetState) :
    messages = [
SystemMessage(content="You are a funny and clever Twitter/X influencer."),
HumanMessage(content=f"""Write a short, original, and hilarious tweet on the topic: "{state['topic']}".
Rules:
- Do NOT use question-answer format.
- Max 280 characters.
- Use observational humor, irony, sarcasm, or cultural references.
- Think in meme logic, punchlines, or relatable takes.
- Use simple, day to day english
""")
]
    tweet = generator_llm.invoke(messages).content
    return {"tweet":tweet}

def evalute_tweet(state :tweetState):
    #prompt
    messages = [
SystemMessage(content="""You are a ruthless, no-laugh-given Twitter critic. You evaluate tweets based on humor,
originality, virality, and tweet format."""),
HumanMessage(content=f"""
Evaluate the following tweet:
Tweet: "{state['tweet']}"
Use the criteria below to evaluate the tweet:
1. Originality - Is this fresh, or have you seen it a hundred times before?
2. Humor - Did it genuinely make you smile, laugh, or chuckle?
3. Punchiness - Is it short, sharp, and scroll-stopping?
4. Virality Potential - Would people retweet or share it?
5. Format - Is it a well-formed tweet (not a setup-punchline joke, not a Q&A joke, and under 280 characters)?
Auto-reject if:
- It's written in question-answer format (e.g., "Why did..." or "What happens when...")
- It exceeds 280 characters
- It reads like a traditional setup-punchline joke
- Dont end with generic, throwaway, or deflating lines that weaken the humor (e.g., "Masterpieces of the auntie-uncle universe" or vague summaries)
### Respond ONLY in structured format:
- evaluation: "approved" or "needs_ improvement"
- feedback: One paragraph explaining the strengths and weaknesses
""")
]
    #evaluate
    result = structured_evaluator.invoke(messages)
    return {"evaluation":result.evaluation, "feedback":result.feedback}

def optimize_tweet(state : tweetState):
    #prompt
    messages = [
SystemMessage(content="You punch up tweets for virality and humor based on given feedback."),
HumanMessage (content=f"""
Improve the tweet based on this feedback: "{state['feedback']}"
Topic: "{state['topic']}"
Original Tweet: {state['tweet']}

Re-write it as a short, viral-worthy tweet. Avoid Q&A style and stay under 280 characters.
""")]
    
    tweet_opt = optimazer_llm.invoke(messages).content
    iteration = state["iteration"] +1 
    return {"tweet" :tweet_opt, "iteration":iteration}

def check_approval(state : tweetState):
    if state['evaluation'] =="approved" or state["iteration"] > state["max_iteration"]:
        return "approved"
    else:
        return "needs_improvment"
    

graph =  StateGraph(tweetState)

graph.add_node("gen_tweet",gen_tweet)
graph.add_node("evaluate_tweet",evalute_tweet)
graph.add_node("optimize_tweet",optimize_tweet)

graph.add_edge(START,"gen_tweet")
graph.add_edge("gen_tweet","evaluate_tweet")
graph.add_conditional_edges("evaluate_tweet",check_approval,{"approved": END, "needs_improvment":"optimize_tweet"})
graph.add_edge("optimize_tweet","evaluate_tweet")
workflow = graph.compile()

initial_state={
    "topic":"kdfjioeqe",
    "iteration":1,
    "max_iteration":3
}

final_state  = workflow.invoke(initial_state)

print(final_state)

