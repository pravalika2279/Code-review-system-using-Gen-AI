from typing import Dict, TypedDict, Optional
from langgraph.graph import StateGraph, END
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

GOOGLE_API_KEY = 'AIzaSyAvWFWZHCj7iR7CHiU_gEnkP9BsEEDhQSw'   
genai.configure(api_key=GOOGLE_API_KEY)

ai_model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

def generate_response(prompt):
    return ai_model.invoke(prompt).content

class ReviewState(TypedDict):
    feedback_notes: Optional[str] = None
    review_history: Optional[str] = None
    source_code: Optional[str] = None
    programming_language: Optional[str] = None
    skill_rating: Optional[str] = None
    iteration_count: Optional[int] = None
    comparison_result: Optional[str] = None
    original_code: Optional[str] = None
    improved_code: Optional[str] = None

code_review_workflow = StateGraph(ReviewState)

review_prompt = "You are a code reviewer proficient in {}. Analyze the given code, adhere to best practices, and identify potential issues. Provide human like feedback as bullet points.\nCode:\n {}"

rewrite_prompt = "You are a developer skilled in {}. Improve the given code by implementing the following recommendations:\n {} \n Code:\n {} \n Output only the modified code."

rating_prompt = "Evaluate the coder's proficiency on a scale of 1 to 10 based on the review cycle and provide a brief reason.\nCode review:\n {} \n"

comparison_prompt = "Examine and rate both the original and revised code snippets on a scale of 1 to 10. Provide separate ratings for each version.\nRevised Code: \n {} \n Original Code: \n {}"

feedback_verification_prompt = "Have all the concerns raised in the feedback been addressed in the updated code? Reply with 'Yes' or 'No'.\nCode: \n {} \n Feedback: \n {} \n"

optimization_prompt = "Enhance the following code for better efficiency and readability while preserving functionality.\nCode:\n {}"

def review_code(state):
    review_history = state.get('review_history', '').strip()
    source_code = state.get('source_code', '').strip()
    programming_language = state.get('programming_language', '').strip()
    iteration_count = state.get('iteration_count')
    
    print("Reviewing code...")
    
    feedback_notes = generate_response(review_prompt.format(programming_language, source_code))
    
    return {'review_history': review_history + "\n REVIEWER:\n" + feedback_notes, 'feedback_notes': feedback_notes, 'iteration_count': iteration_count + 1}

def refine_code(state):
    review_history = state.get('review_history', '').strip()
    feedback_notes = state.get('feedback_notes', '').strip()
    source_code = state.get('source_code', '').strip()
    programming_language = state.get('programming_language', '').strip()
    
    print("Refining code...")
    
    updated_code = generate_response(rewrite_prompt.format(programming_language, feedback_notes, source_code))
    return {'review_history': review_history + '\n DEVELOPER:\n' + updated_code, 'source_code': updated_code}

def optimize_code(state):
    print("Optimizing code...")
    
    improved_code = generate_response(optimization_prompt.format(state.get('source_code', '').strip()))
    return {'improved_code': improved_code}

def finalize_review(state):
    print("Final review in progress...")
    
    review_history = state.get('review_history', '').strip()
    revised_code = state.get('source_code', '').strip()
    original_code = state.get('original_code', '').strip()
    skill_rating = generate_response(rating_prompt.format(review_history))
    
    comparison_result = generate_response(comparison_prompt.format(revised_code, original_code))
    improved_code = generate_response(optimization_prompt.format(revised_code))
    
    return {'skill_rating': skill_rating, 'comparison_result': comparison_result, 'updated_code': revised_code, 'improved_code': improved_code}

code_review_workflow.add_node("review_code", review_code)
code_review_workflow.add_node("refine_code", refine_code)
code_review_workflow.add_node("optimize_code", optimize_code)
code_review_workflow.add_node("finalize_review", finalize_review)

def check_review_completion(state):
    review_done = 1 if 'yes' in generate_response(feedback_verification_prompt.format(state.get('source_code'), state.get('feedback_notes'))).lower() else 0
    max_iterations_reached = 1 if state.get('iteration_count') > 5 else 0
    return "finalize_review" if review_done or max_iterations_reached else "refine_code"

code_review_workflow.add_conditional_edges(
    "review_code",
    check_review_completion,
    {
        "finalize_review": "finalize_review",
        "refine_code": "refine_code"
    }
)

code_review_workflow.set_entry_point("review_code")
code_review_workflow.add_edge('refine_code', "review_code")
code_review_workflow.add_edge('finalize_review', "optimize_code")
code_review_workflow.add_edge('optimize_code', END)

programming_language = input("Enter the programming language (e.g., Python, Java, C++): ")
source_code = input("Enter your code: ")

review_app = code_review_workflow.compile()
review_process = review_app.invoke({"review_history": source_code, "source_code": source_code, 'original_code': source_code, "programming_language": programming_language, 'iteration_count': 0}, {"recursion_limit": 100})

print("HISTORY\n", review_process['review_history'])
print("OPTIMIZED CODE:")
print(review_process['improved_code'])
print("SPECIALIZATION\n", review_process['programming_language'])
print("RATING\n", review_process['skill_rating'])
print("ITERATIONS\n", review_process['iteration_count'])
print("CODE COMPARE\n", review_process['comparison_result'])
print("ACTUAL CODE\n", review_process['original_code'])
