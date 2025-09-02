from langgraph.graph import StateGraph,START,END
from typing import TypedDict

class BMIState(TypedDict):
    weight_lbs : float
    height_in :float
    bmi:float
    category :str

def calculate_bmi(state:BMIState)->BMIState:
    weight_lbs  = state['weight_lbs']
    height_in = state['height_in']
    state['bmi'] = round(703 * weight_lbs / (height_in**2),2)
    return state
def lable_bmi(state:BMIState) -> BMIState:
    bmi = state['bmi']
    if bmi <18.5:
        state['category'] = "underweight"
    elif 18.5<= bmi <= 25:
        state['category'] = "Normal"
    elif 25<=bmi<=30:
        state['category'] = "overweight"
    else:
        state['category'] = "obese"
    return state

graph = StateGraph(BMIState)


graph.add_node('calculate_bmi',calculate_bmi)
graph.add_node('label_bmi',lable_bmi)

graph.add_edge(START,'calculate_bmi')
graph.add_edge('calculate_bmi', 'label_bmi')
graph.add_edge('label_bmi',END)

workflow = graph.compile()

initial_state ={'weight_lbs':150, 'height_in':72}

final_state = workflow.invoke(initial_state)

print(final_state)