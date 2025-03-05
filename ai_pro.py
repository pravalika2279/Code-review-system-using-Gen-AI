import streamlit as st
from typing import Dict, TypedDict, Optional
from langgraph.graph import StateGraph, END
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
# Configure API Key
GOOGLE_API_KEY= "AIzaSyBatkmR41rvYQYVR3SVL9LUadW7PfjHhHU"
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize AI Model
ai_model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

def generate_response(prompt):
    return ai_model.invoke(prompt).content

# Define State Structure
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

# Create Workflow
graph = StateGraph(ReviewState)

# Define Prompts
review_prompt = "You are a code reviewer proficient in {}. Analyze the given code, adhere to best practices, and identify potential issues. Provide human-like feedback as bullet points.\nCode:\n {}"
rewrite_prompt = "You are a developer skilled in {}. Improve the given code by implementing the following recommendations:\n {} \n Code:\n {} \n Output only the modified code."
rating_prompt = "Evaluate the coder's proficiency on a scale of 1 to 10 based on the review cycle and provide a brief reason.\nCode review:\n {} \n"
comparison_prompt = "Examine and rate both the original and revised code snippets on a scale of 1 to 10. Provide separate ratings for original code and optimized code.\nRevised Code: \n {} \n Original Code: \n {}"
feedback_verification_prompt = "Have all the concerns raised in the feedback been addressed in the updated code? Reply with 'Yes' or 'No'.\nCode: \n {} \n Feedback: \n {} \n"
optimization_prompt = "Enhance the following code for better efficiency and readability while preserving functionality. Give the best available code in that functionality.\nCode:\n {}"

# Define Workflow Steps
def review_code(state):
    feedback_notes = generate_response(review_prompt.format(state['programming_language'], state['source_code']))
    return {'review_history': state['review_history'] + "\nREVIEWER:\n" + feedback_notes, 'feedback_notes': feedback_notes, 'iteration_count': state['iteration_count'] + 1}

def refine_code(state):
    updated_code = generate_response(rewrite_prompt.format(state['programming_language'], state['feedback_notes'], state['source_code']))
    return {'review_history': state['review_history'] + '\nDEVELOPER:\n' + updated_code, 'source_code': updated_code}

def optimize_code(state):
    improved_code = generate_response(optimization_prompt.format(state['source_code']))
    return {'improved_code': improved_code}

def finalize_review(state):
    skill_rating = generate_response(rating_prompt.format(state['review_history']))
    comparison_result = generate_response(comparison_prompt.format(state['source_code'], state['original_code']))
    improved_code = generate_response(optimization_prompt.format(state['source_code']))
    return {'skill_rating': skill_rating, 'comparison_result': comparison_result, 'improved_code': improved_code}

# Add Nodes to Workflow
graph.add_node("review_code", review_code)
graph.add_node("refine_code", refine_code)
graph.add_node("optimize_code", optimize_code)
graph.add_node("finalize_review", finalize_review)

def check_review_completion(state):
    review_done = 1 if 'yes' in generate_response(feedback_verification_prompt.format(state['source_code'], state['feedback_notes'])).lower() else 0
    max_iterations_reached = 1 if state['iteration_count'] > 5 else 0
    return "finalize_review" if review_done or max_iterations_reached else "refine_code"

graph.add_conditional_edges("review_code", check_review_completion, {"finalize_review": "finalize_review", "refine_code": "refine_code"})
graph.set_entry_point("review_code")
graph.add_edge('refine_code', "review_code")
graph.add_edge('finalize_review', "optimize_code")
graph.add_edge('optimize_code', END)

# Streamlit UI
st.title("🔍 Code Review System using Generative AI")
st.sidebar.header("Project Settings")
programming_language = st.sidebar.selectbox("Select Programming Language", ["C", "Python", "Java", "C++", "JavaScript", "C#", "Go", "Other"])
source_code = st.text_area("Paste Your Code Here", height=250)

if st.button("Start Review Process"):
    with st.spinner("Reviewing code..."):
        app = graph.compile()
        process = app.invoke({"review_history": source_code, "source_code": source_code, 'original_code': source_code, "programming_language": programming_language, 'iteration_count': 0}, {"recursion_limit": 100})
        
        st.subheader("📜 Review History")
        st.text_area("Review Process", process['review_history'], height=250)

        st.subheader("📂 Original Code")
        st.code(process['original_code'], language=programming_language.lower())
        
        st.subheader("🚀 Optimized Code")
        st.code(process['improved_code'], language=programming_language.lower())
        
        st.subheader("📌 Specialization")
        st.write(process['programming_language'])
        
        st.subheader("⭐ Rating")
        st.write(process['skill_rating'])
        
        st.subheader("🔄 Iterations")
        st.write(process['iteration_count'])
        
        st.subheader("⚖️ Code Comparison")
        st.text_area("Comparison Result", process['comparison_result'], height=150)
        
