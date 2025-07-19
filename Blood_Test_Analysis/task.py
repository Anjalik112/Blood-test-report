from crewai import Task
from agents import doctor, verifier, nutritionist,exercise_agent, abnormal_agent
from tools.tools import BloodTestReportTool, BloodReportSummaryTool, AbnormalInfoSearchTool, NutritionAdviceTool, ExerciseAdviceTool
from typing import List, Optional
from celery_config import celery_app  # Import celery app instance
from time import sleep

# CHANGED:
# Original task definitions were comedic and fictional.
# Rewritten for real medical analysis and production use.

# NEW:
# Helper function to create tasks in a reusable way,
# avoiding repetition and making future changes simpler.
def create_task(
    *,
    description: str,
    expected_output: str,
    agent,
    tools: Optional[List] = None,
) -> Task:
    return Task(
        description=description.strip(),
        expected_output=expected_output.strip(),
        agent=agent,
        tools=tools or [],
        inputs={
            "query": "{query}",
            "report_text": "{report_text}",
        },
        async_execution=False,
    )


# CHANGED:
# Rewritten help_patients task:
# - Original task gave comedic and potentially unsafe advice.
# - Now provides real medical summary and analysis goals.
help_patients = create_task(
    description="""
    You have the patient’s full blood report text:
    
    {report_text}
    
    Follow these steps exactly, and do NOT stop early:
    
    1. Run the `blood_report_summary` tool on the full report text.
    2. Parse the tool result to extract **only those tests labeled as abnormal** (e.g. High, Low, Elevated, Deficiency).
    4. After all tool calls are completed, combine:
        - The `blood_report_summary` result
        
    into a single, **final answer**.

    IMPORTANT:
    - Do NOT treat the `blood_report_summary` output as the final answer by itself.
    - You MUST proceed to `abnormal_info_search` for every abnormal test found.
    - Your final answer must contain:
        - the entire `blood report summary`
        - all article links from `abnormal_info_search`
    """,
    expected_output="A clear, concise patient summary with any relevant article links.",
    agent=doctor,
)
abnormal_task = create_task(
    description="""
    You have 1 input:
     1. The full text output from the `blood_report_summary` tool.
       Do not modify or summarize this text.

    Your Task:
     - Call the `abnormal_info_search` tool exactly once, passing:
        - `summary_text`: the text from the `blood_report_summary` tool 
    - Then produce a single, combined final answer containing:
        1. The **exact text** from the `blood_report_summary` tool (do not alter it).
        2.only the output returned by the `abnormal_info_search` tool.
    - Do not add any extra commentary, narrative, or formatting beyond these two parts.
    """,
    expected_output="A newline-separated list of abnormal test names and their corresponding Mount Sinai Health Library URLs.",
    agent=abnormal_agent,
)


nutrition_analysis = create_task(
    description="""
    You have two inputs:
    
    1. The full text output from the `blood_report_summary` tool.
       Do not modify or summarize this text.
    
    2. The patient's weight in kilograms (numeric value).
    
    Your task:
        
    - Call the `nutrition_advice` tool exactly once, passing:
        - `summary_text`: the text from the `blood_report_summary` tool and `abnormal_info_search` tool.
        - `weight_kg`: the numeric patient weight
    
    - Then produce a single, combined final answer containing:
    
        1. The **exact text** from the `blood_report_summary` tool (do not alter it).
        2. A section titled **“Personalized Nutrition Recommendations:”** followed by
           only the output returned by the `nutrition_advice` tool.
    
    - Do not add any extra commentary, narrative, or formatting beyond these two parts.
    """,
    expected_output=(
        "A single combined output consisting of:\n\n"
        
        "1. A section titled 'Personalized Nutrition Recommendations:' followed only by "
        "the text from the nutrition_advice tool."
    ),
    agent=nutritionist,
)
exercise_routine = create_task(
    description="""
    When a user asks for a wellness routine or daily exercise plan:

    1. Ignore any lab reports or other inputs.
    2. Run the `exercise_advice` tool with no arguments.
    3. Return the full output of `exercise_advice` as the final answer.

    IMPORTANT:
    - Do NOT attempt to parse any blood report or other medical data.
    - Your final answer must be the unmodified suggestions from `exercise_advice`.
    """,
    expected_output="A friendly, comprehensive daily exercise and wellness routine.",
    agent=exercise_agent,
)
# cbc_trend_task = create_task(
#         description="""
#         You have the patient’s full blood report text:
        
#         {report_text}
#         Extract every CBC parameter (with its date) from each report—use the full PDF text, not just the summary—  
#         and then produce a visualization (line or bar chart) showing how key values  
#         (e.g. Hemoglobin, WBC count, Platelet count) change over those dates.
#     """,
#         expected_output="A single time-series chart (or set of charts) displaying trends of CBC parameters across the provided dates.",
#         agent=CBCVisualizer,
#     )

verification = create_task(
    description="""
        Assess whether the provided text appears to be a blood test report.

        Respond with:
        - Yes/No conclusion.
        - Any reasons for your determination.
    """,
    expected_output="Verification result indicating if the input is a blood test report.",
    agent=verifier,
)


# NEW:
# Added Celery task to support asynchronous processing.
# This enables running longer analysis jobs without blocking FastAPI.
@celery_app.task
def process_blood_report(file_path: str, query: str):
    # Simulating a long-running task.
    # Replace with actual call to your analysis pipeline as needed.
    sleep(5)
    print(f"Processing blood test report for file: {file_path} with query: {query}")
    return {"status": "success", "file": file_path, "query": query}