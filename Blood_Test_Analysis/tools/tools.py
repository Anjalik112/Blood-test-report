
## Importing libraries and files
import os
import crewai.tools
from crewai.tools import BaseTool
from pydantic import BaseModel
from typing import Tuple, Dict,List, ClassVar
import re
import requests
import heapq
import logging
import re
import datetime
import matplotlib.pyplot as plt
from crewai.tools import BaseTool
logger = logging.getLogger(__name__)
# PATCH MISSING EnvVar:
# CHANGED:
# Some versions of crewai.tools do not include the EnvVar class,
# which certain tools may depend on (especially when defining tools
# that load environment variables).
# This patch ensures compatibility with CrewAI frameworks.
class EnvVar:
    def __init__(self, name, description=None, required=False):
        self.var_name = name
        self.description = description
        self.required = required

    def get(self):
        return os.getenv(self.var_name)

# Inject the patched EnvVar into crewai.tools
crewai.tools.EnvVar = EnvVar

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import our custom Serper search tool
from tools.serper_dev_tool import SerperDevTool
search_tool = SerperDevTool()

# Import PDF loader from LangChain community library
from langchain_community.document_loaders import PDFPlumberLoader



## Creating custom pdf reader tool

class BloodTestReportTool(BaseTool):
    """
    Custom CrewAI tool for reading blood test reports from PDF files.
    
    CHANGED:
    - Original code had async implementations with missing imports (e.g. PDFLoader).
    - Replaced with PDFPlumberLoader for reliable PDF reading.
    - Introduced MAX_CHARS limit to avoid exceeding LLM context window.
    """

    name: str = "blood_test_report_tool"
    description: str = "Reads data from a PDF file and returns text (truncated to avoid token limits)."

    # Safe maximum character size to avoid LLM token overflows
    MAX_CHARS: int = 3000

    def _run(self, file_path: str) -> str:
        # Load all pages of the PDF
        docs = PDFPlumberLoader(file_path).load()
        full_report = ""

        for data in docs:
            content = data.page_content or ""

            # CHANGED:
            # Remove duplicate newlines for cleaner output
            content = content.replace("\n\n", "\n")
            full_report += content + "\n"

            # Truncate if report grows too large
            if len(full_report) > self.MAX_CHARS:
                full_report = full_report[:self.MAX_CHARS]
                full_report += "\n\n[...TRUNCATED DUE TO SIZE LIMIT...]"
                break

        return full_report

class BloodReportSummaryTool(BaseTool):
    name: str = "blood_report_summary"
    description: str = (
        "Parses every result from a CBC report, shows each value with its reference range, "
        "flags Normal/Low/High, and at the end lists exactly which tests are abnormal along "
        "with their likely clinical implications."
    )

    # Define each test exactly as it appears, with (low, high, unit)
    RANGES: Dict[str, Tuple[float, float, str]] = {
        "Hemoglobin": (13.0, 17.0, "g/dL"),
        "Hemoglobin \(Hb\)": (13.0, 17.0, "g/dL"),
        "Packed Cell Volume \\(PCV\\)": (40.0, 50.0, "%"),
        "RBC Count":                    (4.5, 5.5, "million/mm³"),
        "MCV":                          (83.0, 101.0, "fL"),
        "MCH":                          (27.0, 32.0, "pg"),
        "MCHC":                         (31.5, 34.5, "g/dL"),
        "Red Cell Distribution Width \\(RDW\\)": (11.6, 14.0, "%"),
        "Total Leukocyte Count \\(TLC\\)":       (4.0, 10.0, "thou/mm³"),
        "Segmented Neutrophils":        (40.0, 80.0, "%"),
        "Lymphocytes":                  (20.0, 40.0, "%"),
        "Monocytes":                    (2.0, 10.0, "%"),
        "Eosinophils":                  (1.0, 6.0, "%"),
        "Basophils":                    (0.0, 2.0, "%"),
        "Neutrophils":                  (2.0, 7.0, "thou/mm³"),
        "Lymphocytes":                  (1.0, 3.0, "thou/mm³"),
        "Monocytes":                    (0.2, 1.0, "thou/mm³"),
        "Eosinophils":                  (0.02, 0.50, "thou/mm³"),
        "Basophils":                    (0.02, 0.10, "thou/mm³"),
        "Platelet Count":               (150.0, 410.0, "thou/mm³"),
        "Mean Platelet Volume":         (6.5, 12.0, "fL"),
    }

    # Implications when values are low
    IMPLICATIONS: Dict[str, str] = {
        "Hemoglobin":       "anemia",
        "RBC Count":        "anemia",
        "Total Leukocyte Count (TLC)": "leukopenia",
        "Platelet Count":   "thrombocytopenia",
    }

    def _run(self, report_text: str) -> dict:
        lines = []
        abnormal_details = []

        for test_name, (low, high, unit) in self.RANGES.items():
            m = re.search(rf"{test_name}\s*[:\-]?\s*([\d\.]+)", report_text)
            if not m:
                continue
            val = float(m.group(1))

            if val < low:
                status = "deficiency"
                abnormal_details.append(f"{test_name} → {status}")
                lines.append(f"- {test_name}: {val} {unit} (Ref: {low}–{high} {unit}) → Low (→ suggests {self.IMPLICATIONS.get(test_name,'deficiency')})")
            elif val > high:
                status = "elevation"
                abnormal_details.append(f"{test_name} → {status}")
                lines.append(f"- {test_name}: {val} {unit} (Ref: {low}–{high} {unit}) → High")
            else:
                lines.append(f"- {test_name}: {val} {unit} (Ref: {low}–{high} {unit}) → Normal")

        if abnormal_details:
            summary_line = f"⚠️ Abnormal tests: {'; '.join(abnormal_details)}."
        else:
            summary_line = "✅ All values are within normal ranges."

        return {
            "full_summary": "\n".join(lines),
            "summary_line": summary_line,
            # THIS list still holds the arrows & flags
            "details": abnormal_details,
        }

class AbnormalInfoSearchTool(BaseTool):
    """
    take input from 'details' (which labels abnormalities as 'deficiency' or 'elevation') and returns the
    corresponding Library URL for each.
    Input: full summary text from `details`.
    Output: newline-separated lines of "TestName: URL".
    """
    name: str = "abnormal_info_search"
    description: str = (
       "take input from 'details' (which labels abnormalities as 'deficiency' or 'elevation') " 
        "Library URL for each test."
    )

    # Static mapping of test names to Mount Sinai URLs
    URL_MAP: Dict[str, str] ={
        "Hemoglobin": "https://www.mountsinai.org/health-library/tests/hemoglobin",
        "Hemoglobin \(Hb\)": "https://www.mountsinai.org/health-library/tests/hemoglobin",
        "Hemoglobin (Hb)": "https://www.mountsinai.org/health-library/tests/hemoglobin",
        "Packed Cell Volume (PCV)": "https://labtestsonline.org.uk/tests/pcv",
        "Packed Cell Volume \(PCV\)": "https://labtestsonline.org.uk/tests/pcv",
        "Hematocrit": "https://www.mountsinai.org/health-library/tests/hematocrit",
        "RBC Count": "https://www.mountsinai.org/health-library/tests/rbc-count",
        "Red Blood Cell Count": "https://www.mountsinai.org/health-library/tests/rbc-count",
        "MCV": "https://www.ncbi.nlm.nih.gov/books/NBK545275/",
        "Mean Corpuscular Volume": "https://www.ncbi.nlm.nih.gov/books/NBK545275/",
        "MCH": "https://www.ncbi.nlm.nih.gov/books/NBK545275/",
        "Mean Corpuscular Hemoglobin": "https://www.ncbi.nlm.nih.gov/books/NBK545275/",
        "MCHC": "https://www.medicalnewstoday.com/articles/321303",
        "Mean Corpuscular Hemoglobin Concentration": "https://www.medicalnewstoday.com/articles/321303",
        "RDW": "https://medlineplus.gov/lab-tests/rdw-red-cell-distribution-width/",
        "Red Cell Distribution Width": "https://medlineplus.gov/lab-tests/rdw-red-cell-distribution-width/",
        "Total Leukocyte Count (TLC)": "https://www.ncbi.nlm.nih.gov/books/NBK560882/",
        "White Blood Cell Count": "https://medlineplus.gov/lab-tests/white-blood-count-wbc/",
        "WBC": "https://medlineplus.gov/lab-tests/white-blood-count-wbc/",
        "WBC Differential": "https://www.mountsinai.org/health-library/tests/blood-differential-test",
        "Neutrophils": "https://my.clevelandclinic.org/health/body/22313-neutrophils",
        "Lymphocytes": "https://www.mdanderson.org/cancerwise/what-level-of-lymphocytes-is-considered-dangerous.h00-159701490.html",
        "Monocytes": "https://www.webmd.com/a-to-z-guides/what-to-know-about-high-monocyte-count",
        "Eosinophils": "https://medlineplus.gov/ency/article/003649.htm",
        "Basophils": "https://my.clevelandclinic.org/health/body/23256-basophils",
        "Platelet Count": "https://my.clevelandclinic.org/health/diagnostics/21782-platelet-count",
        "Mean Platelet Volume": "https://medlineplus.gov/lab-tests/mpv-blood-test/",
    }


    def _run(self, details: List[str]) -> str:
        if not details:
            return "No abnormalities found."
        lines = []
        for entry in details:
            # split off the “→ …” to get the test name
            test = entry.split("→")[0].strip()
            url = self.URL_MAP.get(test, "No URL defined")
            lines.append(f"{test}: {url}")
        return "\n".join(lines)


## Creating Nutrition Analysis Tool

class NutritionAdviceTool(BaseTool):
    name: str = "nutrition_advice"
    description: str = (
        "take input from 'details' (which labels abnormalities as 'deficiency' or 'elevation') "
        "returns personalized nutrition advice only for those abnormal tests"
    )

    # Suggestions for deficiencies
    LOW_SUGGESTIONS: ClassVar[Dict[str, str]] = {
    "Hemoglobin": "iron-rich foods (spinach, lentils, red meat, fortified cereals); B₁₂ sources (eggs, dairy, fish); folate (leafy greens, beans)",
    "Hemoglobin \(Hb\)": "iron-rich foods (spinach, lentils, red meat, fortified cereals); B₁₂ sources (eggs, dairy, fish); folate (leafy greens, beans)",
    "Hemoglobin (Hb)": "iron-rich foods (spinach, lentils, red meat, fortified cereals); B₁₂ sources (eggs, dairy, fish); folate (leafy greens, beans)",
    "Packed Cell Volume (PCV)": "iron (beetroot, jaggery); B₁₂ (milk, paneer); folate (green peas, chickpeas)",
    "RBC Count": "iron (dates, ragi); B₁₂ (curd, cheese); folate (asparagus, black-eyed peas)",
    "MCV": "iron (pumpkin seeds); B₁₂ (soy milk, mushrooms); folate (broccoli); selenium & copper (sunflower seeds, cashews)",
    "MCH": "iron-rich foods (amaranth, tofu); vitamin C (citrus fruits, guava) to enhance absorption",
    "MCHC": "iron (dry fruits, sesame); vitamin C (lemon, kiwi) for improved absorption",
    "RDW": "multivitamin with B-complex (whole grains), iron (spinach), copper (dark chocolate, cashews)",
    "Total Leukocyte Count (TLC)": "protein-rich foods (pulses, paneer); vitamins A (carrots), C (amla), E (almonds); zinc (pumpkin seeds)",
    "Segmented Neutrophils": "vitamin A (sweet potato), B₆ (banana, chickpeas), and proteins (eggs, lentils)",
    "Lymphocytes": "vitamin D (sunlight, fortified milk); zinc (flax seeds); vitamin C (capsicum, kiwi)",
    "Monocytes": "lean protein (tofu, legumes); B-complex rich foods (millets, brown rice)",
    "Eosinophils": "anti-inflammatory foods (berries, turmeric); balanced intake of proteins and greens",
    "Basophils": "diverse diet with vegetables, fruits, grains, and nuts to ensure micronutrient sufficiency",
    "Platelet Count": "vitamin K (spinach, kale); B₁₂ (soy products); folate (avocado); protein (dal, milk)",
    "Mean Platelet Volume": "B₁₂ (fortified cereals); folate (cabbage); vitamin B₆ (pistachios, sunflower seeds)",
    "Neutrophils": "Eat protein-rich foods, vitamin B12, folate, and zinc sources to support immune function."
    }


    # Suggestions for elevations
    HIGH_SUGGESTIONS: ClassVar[Dict[str, str]] = {
    "Hemoglobin": "Stay hydrated; limit iron supplements and red meat intake if unnecessary.",
    "Hemoglobin \(Hb\)":"Stay hydrated; limit iron supplements and red meat intake if unnecessary.",
    "Hemoglobin (Hb)":"Stay hydrated; limit iron supplements and red meat intake if unnecessary.",
    "Packed Cell Volume (PCV)": "Increase fluid intake; reduce high-iron foods and alcohol.",
    "RBC Count": "Avoid excess iron; stay hydrated; monitor high-altitude or smoking habits.",
    "MCV": "Check B₁₂ & folate sources; reduce supplements if taken in excess.",
    "MCH": "Moderate iron intake; focus on balanced meals over supplementation.",
    "MCHC": "Hydrate well and reduce intake of iron-fortified foods if elevated.",
    "RDW": "Balance iron and vitamin B intake; avoid excessive multivitamin usage.",
    "Total Leukocyte Count (TLC)": "Avoid inflammatory foods (processed, sugary); include anti-inflammatory items like berries, turmeric.",
    "Segmented Neutrophils": "Reduce stress; include anti-inflammatory foods (green tea, leafy veggies).",
    "Lymphocytes": "Avoid immune stimulants (excess garlic, echinacea); follow a calming, balanced diet.",
    "Monocytes": "Limit red meat; include omega-3 rich foods (flaxseeds, walnuts).",
    "Eosinophils": "Avoid allergens and processed foods; follow an anti-inflammatory diet.",
    "Basophils": "Avoid high-histamine foods (aged cheese, wine); increase water and whole foods.",
    "Platelet Count": "Avoid high-vitamin K foods in excess (e.g., kale, spinach); stay hydrated.",
    "Mean Platelet Volume": "Reduce inflammatory foods; emphasize omega-3s and antioxidant-rich vegetables.",
   "Neutrophils": "Include anti-inflammatory foods like berries, turmeric, leafy greens, and fatty fish; avoid processed and fried foods."
}



    def _run(
        self,
        summary_text: str,      # you can keep this for context or drop it if unused
        details: list[str],     # the list of abnormal strings, e.g. ["Lymphocytes → elevation", ...]
    ) -> str:
        advice_lines: list[str] = []

        # 1️⃣ Protein requirement
        
        advice_lines.append(
            f"• Protein Intake: aim for approximately your weight*1.2 gram per day from eggs, dairy, legumes, and lean meats."
        )

        # 2️⃣ Iterate only over the provided abnormal details
        for entry in details:
            # split into test name and status
            test_name, status_word = [p.strip() for p in entry.split("→", 1)]
            status_key = "deficiency" if "deficiency" in status_word.lower() else "elevation"

            if status_key == "deficiency":
                suggestion = self.LOW_SUGGESTIONS.get(test_name)
                if suggestion:
                    advice_lines.append(f"• {test_name} deficiency: Increase intake of {suggestion}.")
                else:
                    advice_lines.append(f"• {test_name} deficiency: Consult for dietary guidance.")

            else:  # elevation
                suggestion = self.HIGH_SUGGESTIONS.get(test_name)
                if suggestion:
                    advice_lines.append(f"• {test_name} elevation: Dietary changes include {suggestion}.")
                else:
                    advice_lines.append(f"• {test_name} elevation: Consult for dietary guidance.")

        return "\n".join(advice_lines)



## Creating Exercise Planning Tool
class ExerciseAdviceTool(BaseTool):
    """
    Provides general daily exercise and wellness suggestions,
    plus top web and YouTube resources for yoga practice.
    """
    name: str = "exercise_advice"
    description: str = (
        "Suggests daily Pranayama, Yoga, gym workouts, jogging, sports, and mindfulness practices, "
        "and fetches top web & YouTube links for yoga resources."
    )

    def _run(self) -> str:
        suggestions = [
            "• 🧘 Pranayama: 15 minutes daily of alternate nostril breathing (Nadi Shodhana) or Kapalabhati to improve lung function and calm the mind.",
            "• 🧎 Yoga: 20–30 minutes of Sun Salutations (Surya Namaskar), Warrior poses, and gentle stretches to enhance flexibility and strength.",
            "• 🏃 Cardio: 20–30 minutes of jogging, brisk walking, or cycling in the morning to boost cardiovascular health.",
            "• 🏋️ Strength Training: 2–3 sessions per week, focusing on bodyweight exercises (push-ups, squats, lunges) or light gym workouts.",
            "• 🤸‍♂️ Sports & Recreation: 1–2 fun sessions weekly (badminton, basketball, or swimming) for agility and endurance.",
            "• 🧘‍♀️ Meditation: 5–10 minutes of mindfulness or guided meditation each evening to reduce stress and improve sleep quality.",
            "• 💧 Hydration & Rest: Drink 2–3 liters of water daily, and ensure 7–8 hours of quality sleep every night for recovery."
        ]

        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            suggestions.append("\n🔗 External resources not available — SERPER_API_KEY missing.")
            return "\n".join(suggestions)

        headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

        def fetch_one_link(query: str, label: str, retries: int = 2):
            for attempt in range(retries + 1):
                try:
                    resp = requests.post(
                        "https://api.serper.dev/search",
                        headers=headers,
                        json={"q": query, "num": 1},
                        timeout=5,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    items = data.get("organic") or data.get("results") or []
                    if items:
                        url = items[0].get("link") or items[0].get("url")
                        if url:
                            suggestions.append(f"\n🔗 {label}: {url}")
                        return
                except Exception as e:
                    logger.warning(f"[{label}] attempt {attempt + 1} failed: {e}")

        fetch_one_link("top 1 best yoga websites", "Yoga Website")
        fetch_one_link("top 1 best yoga YouTube video", "Yoga YouTube Video")

        return "\n".join(suggestions)

    

class CBCTrendVisualizerTool(BaseTool):
    """
    Visualize time-series trends for key CBC parameters: Hemoglobin, Total Leukocyte Count (TLC), and Platelet Count.
    Input: a list of dicts, each with:
        - date: ISO-format date string (YYYY-MM-DD)
        - report_text: full text of a CBC report
    Output: path to a generated trend chart image
    """
    name: str = "cbc_key_trend_visualizer"
    description: str = (
        "Visualize trends over time for Hemoglobin, WBC count, and Platelet count."
    )

    PARAMETERS: List[str] = [
        "Hemoglobin",
        "Total Leukocyte Count \(TLC\)",
        "Platelet Count",
    ]

    # Reference ranges (optional, only for bar shading if needed)
    RANGES: Dict[str, tuple] = {
        "Hemoglobin": (13.0, 17.0),
        "Total Leukocyte Count (TLC)": (4.0, 10.0),
        "Platelet Count": (150.0, 410.0),
    }

    PATTERNS: Dict[str, re.Pattern] = {
        param: re.compile(rf"{re.escape(param)}\s*[:\-]?\s*([\d\.]+)", re.IGNORECASE)
        for param in PARAMETERS
    }

    def _run(self, reports: List[Dict[str, str]]) -> str:
        # Parse dates and values
        dates: List[datetime.date] = []
        series: Dict[str, List[float]] = {p: [] for p in self.PARAMETERS}

        for entry in reports:
            # parse date
            date_str = entry.get("date", "").split()[0]
            try:
                dt = datetime.datetime.fromisoformat(date_str).date()
            except ValueError:
                dt = datetime.datetime.strptime(date_str, "%d/%m/%Y").date()
            dates.append(dt)
            text = entry.get("report_text", "")
            for param, pattern in self.PATTERNS.items():
                m = pattern.search(text)
                series[param].append(float(m.group(1)) if m else None)

        # Sort by date
        sorted_idx = sorted(range(len(dates)), key=lambda i: dates[i])
        dates_sorted = [dates[i] for i in sorted_idx]
        values_sorted = {param: [series[param][i] for i in sorted_idx] for param in self.PARAMETERS}

        # Plot
        plt.figure(figsize=(10, 6))
        for param, vals in values_sorted.items():
            plt.plot(dates_sorted, vals, marker='o', label=param)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Key CBC Parameter Trends')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        out_path = '/mnt/data/cbc_key_trend.png'
        plt.savefig(out_path)
        plt.close()
        return out_path



__all__ = [
    "BloodTestReportTool",
    "AbnormalInfoSearchTool",
    "BloodReportSummaryTool",
    "NutritionTool",
    "ExerciseTool",
    "search_tool",
]