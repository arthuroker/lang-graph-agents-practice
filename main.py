from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
import os

load_dotenv()

llm = init_chat_model(
  model_provider="groq",
  model="gemma2-9b-it",
)

class MessageClassifier(BaseModel):
  message_type: Literal["email", "code"] = Field(
    ...,
    description="Classify if the message requires email or code feedback"
  )

class State(TypedDict):
  messages: Annotated[list, add_messages]
  message_type: str | None

def classify_message(state: State):
  last_message = state["messages"][-1]
  classifier_llm = llm.with_structured_output(MessageClassifier)

  result = classifier_llm.invoke([
    {
      "role":"system",
      "content": 
    """Role: You are an LLM designed to classify messages as 'email' or 'code'.

    Task: You **must** classify the user's message. To do this, you will use the 'MessageClassifier' tool.
    The 'message_type' parameter for this tool **must** be one of the following exact string values: "email" or "code".
    Do not use any other value for 'message_type'.

    Classification Guidelines:
    1. Choose 'email': If the message resembles an email, typically containing formal language, salutations (e.g., Dear, Hi), closings (e.g., Sincerely, Best regards), and phrases like 'I hope all is well'.
    2. Choose 'code': If the message primarily consists of programming code, regardless of the programming language. Code often involves line-by-line instructions, special characters, and keywords specific to programming.

    Carefully analyze the user's message and then call the 'MessageClassifier' tool with the determined 'message_type'.
    """
      
    },
    {"role":"user", "content": last_message.content}
  ])

  return {"message_type": result.message_type}

def router(state: State):
  message_type = state.get("message_type", "email")
  if message_type == "code":
    return {"next": "code"}
  
  return {"next": "email"}

def email_agent(state: State):
  last_message = state["messages"][-1]

  messages = [
    {"role": "system",
     "content": """Situation
      You are a professional email editor working with a client who values conciseness, clarity, and maintaining their authentic voice. The client regularly drafts emails that need refinement while preserving their personal style, including their standard "Sincerely" sign-off.

      Task
      Review and enhance email drafts the client provides. Your edits should:

      Dramatically increase conciseness by eliminating unnecessary words
      Improve clarity by restructuring confusing sentences
      Maintain the client's authentic voice and personality
      Preserve the "Sincerely" sign-off
      Infuse the precision and power of Robert Greene's writing style
      Objective
      Transform ordinary email drafts into powerful, clear communications that command attention and respect while remaining true to the client's voice. The final product should achieve maximum impact with minimum words.

      Knowledge
      Robert Greene's writing style is characterized by:

      Directness and precision in language
      Strategic use of brevity for impact
      Clear, authoritative tone
      Elimination of superfluous words
      Powerful, declarative statements
      Psychological insight conveyed through concise phrasing
      Your life depends on ruthlessly eliminating unnecessary words while preserving the core message and the client's authentic voice. Never add flowery language or unnecessary elaboration. When in doubt, cut. """},
     { "role": "user",
      "content": last_message.content}
  ]

  reply = llm.invoke(messages)
  return {"messages": [{"role": "assistant", "content": reply.content}]}

def code_agent(state: State):
  last_message = state["messages"][-1]

  messages = [
    {"role": "system",
     "content": """Situation
      You are an expert coding agent with deep knowledge of software engineering best practices, design patterns, and modern coding standards across multiple programming languages. You're tasked with refactoring code while preserving its core functionality.

      Task
      Analyze the provided code and refactor it according to these specific requirements:

      Maintain identical logical functionality and behavior
      Apply modern, standard coding practices appropriate for the language
      Add clear, concise comments at crucial sections
      Ensure proper indentation, alignment, and consistent spacing
      Improve code readability and maintainability
      Objective
      Transform legacy or suboptimal code into high-quality, production-ready code that follows industry best practices while preserving its original functionality, making it more maintainable, readable, and aligned with modern development standards.

      Knowledge

      Apply language-specific conventions and idioms
      Implement appropriate design patterns when beneficial
      Use meaningful variable/function names that clearly indicate purpose
      Break down complex functions into smaller, more manageable pieces
      Remove redundant or duplicate code
      Optimize inefficient algorithms or data structures when possible
      Ensure consistent formatting throughout (spacing, indentation, line breaks)
      Your life depends on maintaining the exact same functionality while improving the code's structure and readability
      When you've completed the refactoring, explain the key improvements you've made and how they benefit the codebase while maintaining the original functionality."""},
     { "role": "user",
      "content": last_message.content}
  ]

  reply = llm.invoke(messages)
  return {"messages": [{"role": "assistant", "content": reply.content}]}

graph_builder = StateGraph(State)

graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("code_agent", code_agent)
graph_builder.add_node("email_agent", email_agent)

graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")

graph_builder.add_conditional_edges(
  "router",
  lambda state: state.get("next"),
  {"code": "code_agent", "email": "email_agent"}
)

graph_builder.add_edge("code_agent", END)
graph_builder.add_edge("email_agent", END)


graph = graph_builder.compile()

def run_chatbot():
  state = {"messages": [], "message_type": None}

  while True:
    user_input = input("Message: ")
    if user_input == "exit":
      print("Bye")
      break

    state["messages"] = state.get("messages", []) + [
      {"role": "user", "content": user_input}
    ]

    state = graph.invoke(state)

    if state.get("messages") and len(state["messages"]) > 0:
      last_message = state["messages"][-1]
      print(f"Assistant: {last_message.content}")

if __name__ == "__main__":
  run_chatbot()