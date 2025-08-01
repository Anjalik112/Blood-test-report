
from crewai import Crew
from task import help_patients, nutrition_analysis, exercise_routine, abnormal_task, verification
from agents import doctor, nutritionist, exercise_agent, abnormal_agent, verifier
from Blood_Test_Analysis.tools.tools import (
    BloodTestReportTool,
    BloodReportSummaryTool,
    AbnormalInfoSearchTool,
    NutritionAdviceTool,
    ExerciseAdviceTool
)
import logging
import time

logger = logging.getLogger(__name__)

def select_reports_by_query(query: str, outputs: dict) -> dict:
    """
    Return only the pieces of `outputs` that make sense for this query.
    - summary requests → all four.
    - range / why‑is‑my‑Hb‑low questions → doctor + abnormal.
    - nutrition/meal questions → nutrition only.
    - exercise questions → exercise only.
    - fallback → doctor + abnormal.
    """
    q = query.lower()

    if any(kw in q for kw in ("summary", "summarise", "overview", "all results")):
        return {
            "doctor_report":    outputs["doctor_report"],
            "abnormal_info":    outputs["abnormal_info"],
            "nutrition_plan":   outputs["nutrition_plan"],
            "exercise_routine": outputs["exercise_routine"],
        }

    # 2) blood‑range / why low‑hb
    if any(kw in q for kw in ("range", "why my hb", "hemoglobin", "low", "high", "elevation", "deficiency")):
        return {
            "doctor_report": outputs["doctor_report"],
            "abnormal_info": outputs["abnormal_info"],
        }

    # 3) nutrition only
    if any(kw in q for kw in ("nutrition", "meal", "diet", "nutrient", "food plan")):
        return {"nutrition_plan": outputs["nutrition_plan"]}

    # 4) exercise only
    if any(kw in q for kw in ("exercise", "workout", "routine", "physical activity")):
        return {"exercise_routine": outputs["exercise_routine"]}

    # 5) fallback to doctor + abnormal
    return {
        "doctor_report": outputs["doctor_report"],
        "abnormal_info": outputs["abnormal_info"],
    }


def run_crew_pipeline(query: str, file_path: str) -> dict:
    """
    Runs the CrewAI pipeline and returns a dict with selected report sections
    based on the query intent.
    """
    # Instantiate the custom PDF reading tool
    tool = BloodTestReportTool()

    try:
        pdf_text = tool.run(file_path)
    except AttributeError:
        pdf_text = tool._run(file_path)

    # Prepare inputs for the CrewAI pipeline
    inputs = {
        "query": query,
        "report_text": pdf_text,
    }

    # Instantiate the CrewAI pipeline
    crew = Crew(
        agents=[doctor, abnormal_agent, nutritionist, exercise_agent],
        tasks=[help_patients, abnormal_task, nutrition_analysis, exercise_routine],
        process="sequential",
    )
    # Default outputs
    outputs = {
        "doctor_report":    "No doctor analysis generated.",
        "abnormal_info":    "No abnormal info generated.",
        "nutrition_plan":   "No nutrition plan generated.",
        "exercise_routine": "No exercise routine generated.",
    }

    try:
        time.sleep(1)
        results = crew.kickoff(inputs)
        result_dict = results.dict()
        tasks = result_dict.get("tasks_output", [])

        # Assign in order
        if len(tasks) > 0 and tasks[0].get("raw"):
            outputs["doctor_report"] = tasks[0]["raw"].strip()
        if len(tasks) > 1 and tasks[1].get("raw"):
            outputs["abnormal_info"] = tasks[1]["raw"].strip()
        if len(tasks) > 2 and tasks[2].get("raw"):
            outputs["nutrition_plan"] = tasks[2]["raw"].strip()
        if len(tasks) > 3 and tasks[3].get("raw"):
            outputs["exercise_routine"] = tasks[3]["raw"].strip()

    except Exception as e:
        logger.exception("Error running crew pipeline.")
        outputs["doctor_report"] = f"Error during analysis: {e}"

    # Filter outputs based on query intent
    filtered = select_reports_by_query(query, outputs)
    return filtered
