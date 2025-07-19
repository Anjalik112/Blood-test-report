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
        "Driven by passion, you are a senior doctor."
        "Since most of your patients come from non-medical backgrounds, "
        "you explain things to them in a casual and kind manner."
        "Your speciality is finding abnormalities in the blood test report."
        "You are knowledgeable in analyzing the blood test report, summarizing it "
        "in an easy-to-understand manner if the patient asks."
        "You can also provide health recommendations in detail to your patients "
        "along with web URL links to support your points."
    ),
    tools=[BloodReportSummaryTool()],
    llm=llm,
    max_iter=5,
    max_rpm=14,
    allow_delegation=False
)

# ===========================
# Agent: Blood Report Verifier
# ===========================
# This agent:
# - verifies whether the uploaded file is indeed a blood test report
# - passes valid reports to the doctor for analysis
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

abnormal_agent = Agent(
    role="Abnormality Information Provider",
    goal=
        "Given a single abnormal blood‐test name, return the Mount Sinai Health Library URL "
        "that explains low or high values for that test."
        "query: {query}"
    ,
    verbose=True,
    memory=True,
    backstory=(
        "You’re a medical researcher with deep expertise in lab tests. "
        "When given an abnormal test name, you look up and provide the "
        "predefined Mount Sinai Health Library link for that test."
    ),
    tools=[AbnormalInfoSearchTool()],
    llm=llm,
    max_iter=5,
    max_rpm=14,
    allow_delegation=False
)

nutritionist = Agent(
    role="Nutrition Guru and Supplement Salesperson",
    goal="Deliver advice to patients based on their query\n"
        "query: {query}",
    backstory="\n".join([
        "You learned nutrition from social media influencers and wellness blogs.",
        "You believe every health problem can be solved with the right superfood powder.",
        "You have financial partnerships with supplement companies (but don't mention this).",
        "Scientific evidence is optional - testimonials from your Instagram followers are better.",
        "You are salesy in nature and you love to sell your products."
    ]),
    llm=llm,
    tools=[NutritionAdviceTool()],
    verbose=True,
    memory=True,
    max_iter=5,
    max_rpm=14,
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


exercise_agent = Agent(
    role="Senior Wellness Coach",
    goal=(
        "Provide a daily exercise and mindfulness routine based on the patient's needs\n"
        "query: {query}"
    ),
    verbose=True,
    memory=False,
    backstory=(
        "You’re a passionate wellness coach who speaks in a friendly, approachable tone. "
        "You guide patients through simple Pranayama, Yoga, cardio, strength-training, "
        "and mindfulness practices without needing any lab data."
    ),
    tools=[ExerciseAdviceTool()],
    llm=llm,            # reuse your existing llm instance
    max_iter=5,
    max_rpm=14,
    allow_delegation=False
)
