import signal

# ---- Windows compatibility patch for CrewAI ----
if not hasattr(signal, "SIGHUP"):
    signal.SIGHUP = 1
if not hasattr(signal, "SIGCONT"):
    signal.SIGCONT = 2
if not hasattr(signal, "SIGTSTP"):
    signal.SIGTSTP = signal.SIGTERM



import os
import asyncio

from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI


# ---------------------------
# Windows asyncio fix
# ---------------------------
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# ---------------------------
# Initialize Gemini LLM
# ---------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    # convert_system_message_to_human=True  # IMPORTANT for Gemini
)


# ---------------------------
# Define Agents
# ---------------------------
research_agent = Agent(
    role="Research Analyst",
    goal="Research the given topic and extract key insights",
    backstory=(
        "You are an expert research analyst skilled at breaking down complex topics "
        "into concise, actionable insights."
    ),
    llm=llm,
    verbose=True
)

writer_agent = Agent(
    role="Technical Writer",
    goal="Create a clear and well-structured summary from research insights",
    backstory=(
        "You are a professional technical writer who explains complex ideas "
        "in a simple and structured manner."
    ),
    llm=llm,
    verbose=True
)


# ---------------------------
# Define Tasks
# ---------------------------
research_task = Task(
    description=(
        "Research the topic: 'Agentic AI frameworks'. "
        "Focus on CrewAI, LangGraph, and AutoGen. "
        "Provide key features, use cases, and limitations."
    ),
    expected_output=(
        "A bullet-point list of insights covering features, use cases, "
        "and limitations of each framework."
    ),
    agent=research_agent
)

writing_task = Task(
    description=(
        "Using the research insights, write a structured summary "
        "comparing CrewAI, LangGraph, and AutoGen."
    ),
    expected_output=(
        "A clear comparison summary with headings and concise explanations."
    ),
    agent=writer_agent
)


# ---------------------------
# Define Crew
# ---------------------------
crew = Crew(
    agents=[research_agent, writer_agent],
    tasks=[research_task, writing_task],
    process=Process.sequential,
    verbose=True
)


# ---------------------------
# Run Crew (IMPORTANT for Windows)
# ---------------------------
if __name__ == "__main__":
    result = crew.kickoff()
    print("\n================ FINAL OUTPUT ================\n")
    print(result)
