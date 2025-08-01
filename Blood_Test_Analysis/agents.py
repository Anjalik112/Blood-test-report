import os
from dotenv import load_dotenv
load_dotenv()
from tools.tools import BloodTestReportTool, BloodReportSummaryTool, AbnormalInfoSearchTool, NutritionAdviceTool, ExerciseAdviceTool
import time
from crewai import LLM
from langchain_groq import ChatGroq
from crewai import Agent

# Load Groq API key from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize the Groq LLM
llm = ChatGroq(
    temperature=0.7,
    model_name="groq/gemma2-9b-it",
    api_key=GROQ_API_KEY
)

doctor = Agent(
    role="Senior Experienced Doctor",
    goal=(
        "Deliver advice to patients based on their query\n"
        "query: {query}"
    ),
    verbose=True,
    memory=True,
    backstory=(
    "You're a highly experienced doctor known for making complex medical reports easy to understand. "
    "You receive structured data from tools like lab result summaries. "
    "You don’t just repeat those findings — you explain what they mean for the patient's health, "
    "what possible causes might exist, and what they should discuss with their physician. "
    "You write in a clear, kind, and slightly conversational tone so patients feel comforted and informed."
    ),
    tools=[BloodReportSummaryTool()],
    llm=llm,
    max_iter=5,
    max_rpm=14,
    allow_delegation=True
)


abnormal_agent = Agent(
    role="Abnormality Information Provider",
    goal=
        "Given a single abnormal blood‐test name, return the Health Library URL "
        "that explains low or high values for that test."
        "query: {query}"
    ,
    verbose=True,
    memory=True,
    backstory=(
        "You’re a medical researcher with deep expertise in lab tests. "
        "When given an abnormal test name, you look up and provide the "
        "predefined Health Library link for that test."
    ),
    tools=[AbnormalInfoSearchTool()],
    llm=llm,
    max_iter=4,
    max_rpm=14,
    allow_delegation=False
)

nutritionist = Agent(
    role="Nutrition Guru",
    goal="Deliver advice to patients based on their query\n"
        "query: {query}",
    backstory="\n".join([
        "You learned nutrition from nutrition course and wellness blogs.",
        "You believe every health problem can be solved with the right superfood.",
        "Scientific evidence is optional - testimonials.",
        "you are expert in homemade food suggetions."
    ]),
    llm=llm,
    tools=[NutritionAdviceTool()],
    verbose=True,
    memory=True,
    max_iter=4,
    max_rpm=14,
    allow_delegation=False
)

exercise_agent = Agent(
    role="Senior Wellness Coach",
    goal=(
        "Provide a daily exercise and mindfulness routine based on the patient's needs\n"
        "query: {query}"
    ),
    verbose=True,
    memory=True,
    backstory=(
        "You’re a passionate wellness coach who speaks in a friendly, approachable tone. "
        "You guide patients through simple Pranayama, Yoga, cardio, strength-training, "
        "and mindfulness practices without needing any lab data."
    ),
    tools=[ExerciseAdviceTool()],
    llm=llm,            # reuse your existing llm instance
    max_iter=7,
    max_rpm=14,
    allow_delegation=False
)


verifier = Agent(
    role="Blood Report Verifier",
    goal=(
        "Only read the data once.\n"
        "You will be provided with a path to the file given by the user, "
        "read the data of the file provided by the user and use your knowledge "
        "to verify if the data is a blood report or not.\n"
    ),
    verbose=True,
    memory=True,
    backstory=(
        "You have experience with understanding a blood report in any format. "
        "You always read the blood report and then pass it to the senior doctor after verifying it."
    ),
    tools=[BloodTestReportTool()],
    llm=llm,
    max_iter=1,
    max_rpm=7,
    allow_delegation=False
)

# CBCVisualizer = Agent(
#     role="Cbc Visulaizer tool for better understanding of reports",
#     backstory="\n".join([
#         "When you weren’t coding new chart libraries in your spare time, you were in triage rooms watching trends save patients.",
#         "You combine your two passions—beautiful, interactive plots and rock-solid medical insights—to give clinicians an instant “big-picture” view of blood health.",
#     ]),
#     goal="Given a series of past CBC PDFs, extract each parameter and produce an interactive time-series chart of hemoglobin, WBC, platelets, etc., highlighting any highs or lows.\n"
#     "query: {query}",
#     llm=llm,
#     tools=[CBCTrendVisualizerTool()],
#     verbose=True,
#     memory=True,
#     max_iter=10,
#     max_rpm=7,
#     allow_delegation=False
# )