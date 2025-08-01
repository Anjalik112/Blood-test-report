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
You are provided with a patient’s full blood test report:

{report_text}

Steps to follow:

1. Call the `blood_report_summary` tool on the report text.  
2. From its output, extract:
   - `full_summary` (the line‑by‑line CBC results with Normal/High/Low flags)  
   - `details` (the list of strings for each abnormal test, e.g. ["Lymphocytes → elevation", …])  
3. Construct your final answer exactly in this structure:
   a) **Full CBC Summary:**  
      <insert full_summary from the tool>  
   b) **⚠️ Abnormalities Detected:**  
      <comma‑separated list of test names from `details`, always including the flags (e.g., "Lymphocytes → elevation")>  
   c) **🩺 Interpretation (1–2 lines):**  
      A brief explanation of what these abnormalities collectively suggest (e.g., possible infection, inflammation, allergy, or nutritional deficiency).

Tone:
- Kind, simple, and conversational as if speaking to a non‑medical patient.
- Begin with “Dear `user_name`,” and end with a gentle suggestion to discuss with their physician.

IMPORTANT:
- Do NOT re‑parse the text yourself—use exactly the outputs from `blood_report_summary`.
- Do NOT include anything beyond the three sections above.
- Ensure the abnormalities list **always retains both the test name and its flag** as returned.
""",
    expected_output=(
        "Dear `user_name`,\n\n"
        "🩸Full CBC Summary:\n"
        "<full_summary from blood_report_summary>\n\n"
        "⚠️Abnormalities Detected:\n"
        "`details` (include test names with flags)\n\n"
        "🩺Interpretation:\n"
        "<1–2 sentence explanation>\n"
    ),
    agent=doctor,
)



abnormal_task = create_task(
    description="""
You are given:
- `details`: a list of abnormal test detail strings (e.g., "Lymphocytes → elevation").

Your task:

1. Begin with the heading: `⚠️Abnormalities Detected:` followed by a friendly 1-liner.

2. For each entry in `details` (which is always a **string**, not a dictionary), extract:
   - the test name
   - the abnormality type (e.g., elevation or deficiency)

3. For each test:
   - Generate a bullet point in the format: `- <TestName> (<flag>): <brief explanation>`
   - Use clinical reasoning (in simple, non-technical words) to explain the significance of the abnormality.

4. Then, call `abnormal_info_search(details=...)` with the exact same string list.
   - This tool returns a list of `TestName: URL` mappings.
   - Paste the tool's output **verbatim**, under the heading: `🔗 Resources for Understanding`

⚠️ Rules:
- The `details` input will **always be a list of strings**, like `"Neutrophils → elevation"`, **not dictionaries**.
- Do **not** modify or format the tool's output.
- Do **not** include the raw `details` list in the final answer.
- The output must feel like a helpful summary followed by reference links.
""",
    expected_output=(
        "⚠️Abnormalities Detected:\n"
        "- <TestName> (<flag>): <brief explanation>\n"
        "- ...\n\n"
        "🔗 Resources for Understanding:\n"
        "<abnormal_info_search tool's output>\n\n"
    ),
    agent=abnormal_agent,
)





nutrition_analysis = create_task(
    description="""
You have two inputs:

1. `full_summary`: the exact text output from the `blood_report_summary` tool.  
2. `details`: a list of abnormal test detail strings from `blood_report_summary`.  

Your task:

1. Call the `nutrition_advice` tool exactly once, passing:
   - `details`: the list of abnormal test detail strings  

2. Then produce a single, combined final answer containing **exactly** these sections, in this order:

---

🥗 Personalized Nutrition Recommendations: 
<insert the output returned by `nutrition_advice`>  

🧃 General Wellness Tips:
End with 1–2 friendly, general health tips not specific to the report — such as hydration, seasonal fruits, or reducing processed foods.  
These should be short, practical, and helpful for most people.
""",
    expected_output=(
        "🥗 Personalized Nutrition Recommendations:\n"
        "<nutrition_advice tool's output>\n\n"
        "🧃 General Wellness Tips:\n"
        "<1–2 friendly, general health suggestions>"
    ),
    agent=nutritionist,
)


exercise_routine = create_task(
    description="""
When a user asks for a daily exercise or wellness routine:

1. Ignore any lab reports or other medical inputs.
2. Call the `exercise_advice` tool exactly once, with no arguments.
   - Note: this tool will also fetch top yoga websites and YouTube videos via the Serper API if `SERPER_API_KEY` is set.
3. Present its output verbatim under a clear, emoji‑headed section.

Format your final answer exactly as follows:

🏋️‍♂️ Daily Routine & Wellness Plan**  
<unmodified output from `exercise_advice` — including any “top 1 Yoga Practice Websites” or “Top 1 Yoga Practice YouTube Videos” sections>

Important:
- Do NOT add, remove, or rephrase any lines.
- Preserve bullet points, emojis, and any external‑links sections intact.
""",
    expected_output=(
        "🏋️‍♂️ Daily Routine & Wellness Plan\n"
        "<full, verbatim output of the exercise_advice tool, including Serper‑fetched links if available>"
    ),
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

@celery_app.task
def process_blood_report(file_path: str, query: str):
    # Simulating a long-running task.
    # Replace with actual call to your analysis pipeline as needed.
    sleep(5)
    print(f"Processing blood test report for file: {file_path} with query: {query}")
    return {"status": "success", "file": file_path, "query": query}