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


## SCHEMA FOR BLOOD TEST REPORT TOOL
# NEW:
# Defines a schema for validating inputs to the BloodTestReportTool.
# Currently unused in the tool itself, but could be useful for
# future integrations with CrewAI or FastAPI validation.
class BloodTestReportToolSchema(BaseModel):
    file_path: str


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

    def _run(self, report_text: str) -> str:
        lines: List[str] = []
        abnormal_details: List[str] = []

        for test_name, (low, high, unit) in self.RANGES.items():
            pattern = rf"{test_name}\s*[:\-]?\s*([\d\.]+)"
            m = re.search(pattern, report_text, re.IGNORECASE)
            if not m:
                continue

            val = float(m.group(1))
            ref = f"{low}–{high} {unit}"

            if val < low:
                imp = self.IMPLICATIONS.get(test_name, "deficiency")
                status = f"Low (→ suggests {imp})"
                abnormal_details.append(f"{test_name} → {imp}")
            elif val > high:
                status = "High"
                abnormal_details.append(f"{test_name} → elevation")
            else:
                status = "Normal"

            lines.append(f"- {test_name}: {val} {unit} (Ref: {ref}) → {status}")

        # Build the final summary
        if not abnormal_details:
            summary = "✅ All values are within normal ranges."
        else:
            # List each abnormal test with its implication
            details = "; ".join(abnormal_details)
            summary = (
                f"⚠️ Abnormal tests: {details}. "
                "Please consult your physician for further evaluation."
            )

        return "\n".join(lines) + "\n\n" + summary

class AbnormalInfoSearchTool(BaseTool):
    """
    Parses abnormal test names from a blood report summary and returns the
    corresponding Mount Sinai Health Library URL for each.
    Input: full summary text from `blood_report_summary`.
    Output: newline-separated lines of "TestName: URL".
    """
    name: str = "abnormal_info_search"
    description: str = (
       "Given the output from 'blood_report_summary' (which labels abnormalities as 'deficiency' or 'elevation') "
        "Library URL for each test."
    )

    # Static mapping of test names to Mount Sinai URLs
    URL_MAP: Dict[str, str] = {
        "Hemoglobin": "https://www.mountsinai.org/health-library/tests/hemoglobin",
        "Hemoglobin (Hb)": "https://www.mountsinai.org/health-library/tests/hemoglobin",
        "Packed Cell Volume (PCV)": "https://www.mountsinai.org/health-library/tests/hematocrit",
        "Hematocrit": "https://www.mountsinai.org/health-library/tests/hematocrit",
        "RBC Count": "https://www.mountsinai.org/health-library/tests/red-blood-cell-count",
        "Red Blood Cell Count": "https://www.mountsinai.org/health-library/tests/red-blood-cell-count",
        "MCV": "https://www.mountsinai.org/health-library/tests/mean-corpuscular-volume-mcv",
        "Mean Corpuscular Volume": "https://www.mountsinai.org/health-library/tests/mean-corpuscular-volume-mcv",
        "MCH": "https://www.mountsinai.org/health-library/tests/mean-corpuscular-hemoglobin-mch",
        "Mean Corpuscular Hemoglobin": "https://www.mountsinai.org/health-library/tests/mean-corpuscular-hemoglobin-mch",
        "MCHC": "https://www.mountsinai.org/health-library/tests/mean-corpuscular-hemoglobin-concentration-mchc",
        "Mean Corpuscular Hemoglobin Concentration": "https://www.mountsinai.org/health-library/tests/mean-corpuscular-hemoglobin-concentration-mchc",
        "RDW": "https://www.mountsinai.org/health-library/tests/red-cell-distribution-width-rdw",
        "Red Cell Distribution Width": "https://www.mountsinai.org/health-library/tests/red-cell-distribution-width-rdw",
        "Total Leukocyte Count (TLC)": "https://www.mountsinai.org/health-library/tests/white-blood-cell-count-wbc",
        "White Blood Cell Count": "https://www.mountsinai.org/health-library/tests/white-blood-cell-count-wbc",
        "WBC": "https://www.mountsinai.org/health-library/tests/white-blood-cell-count-wbc",
        "WBC Differential": "https://www.mountsinai.org/health-library/tests/white-blood-cell-differential",
        "Neutrophils": "https://www.mountsinai.org/health-library/tests/white-blood-cell-differential",
        "Lymphocytes": "https://www.mountsinai.org/health-library/tests/white-blood-cell-differential",
        "Monocytes": "https://www.mountsinai.org/health-library/tests/white-blood-cell-differential",
        "Eosinophils": "https://www.mountsinai.org/health-library/tests/white-blood-cell-differential",
        "Basophils": "https://www.mountsinai.org/health-library/tests/white-blood-cell-differential",
        "Platelet Count": "https://www.mountsinai.org/health-library/tests/platelet-count",
        "Mean Platelet Volume": "https://www.mountsinai.org/health-library/tests/mean-platelet-volume-mpv",
    }

    def _run(self, summary_text: str) -> str:
        # Extract the "Abnormal tests: ..." portion
        match = re.search(r"Abnormal tests:\s*(.+?)(?:\.|$)", summary_text, re.IGNORECASE)
        if not match:
            return "No abnormalities found."

        # Split out the test names
        raw_entries = re.split(r"[;,]", match.group(1))
        test_names = [e.strip().split("→")[0].strip() for e in raw_entries if e.strip()]

        # Build results using the static map
        lines = []
        for test in test_names:
            url = self.URL_MAP.get(test)
            if not url:
                url = "No URL defined for this test"
            lines.append(f"{test}: {url}")

        return "\n".join(lines)



   
## Creating Nutrition Analysis Tool

class NutritionAdviceTool(BaseTool):
    name: str = "nutrition_advice"
    description: str = (
        "Given the output from 'blood_report_summary' (which labels abnormalities as 'deficiency' or 'elevation') "
        "and the patient’s weight in kg, returns personalized nutrition advice only for those abnormal tests, plus protein requirements."
    )

    # Suggestions for deficiencies
    LOW_SUGGESTIONS: ClassVar[Dict[str, str]] = {
        "Hemoglobin": "iron-rich foods (spinach, lentils, red meat, fortified cereals); B₁₂ sources (eggs, dairy, fish); folate (leafy greens, beans)",
        "Packed Cell Volume (PCV)": "same as Hemoglobin",
        "RBC Count": "same as Hemoglobin",
        "MCV": "iron, B₁₂ & folate; selenium & copper for RBC maturation",
        "MCH": "iron-rich foods; vitamin C to enhance absorption",
        "MCHC": "iron and vitamin C supplements",
        "RDW": "multivitamin with B-complex, iron, copper",
        "Total Leukocyte Count (TLC)": "protein-rich diet (lean meats, pulses); vitamins A/C/E; zinc (nuts, seeds)",
        "Segmented Neutrophils": "vitamin A (carrots, sweet potato); B₆ (chickpeas); protein support",
        "Lymphocytes": "vitamin D (sunlight, fortified milk); zinc (pumpkin seeds); vitamin C",
        "Monocytes": "balanced diet with B-complex vitamins; lean protein",
        "Eosinophils": "focus on overall nutrient balance; no specific foods",
        "Basophils": "typical diet sufficient; focus on variety",
        "Platelet Count": "vitamin K (leafy greens, broccoli); B₁₂ & folate; protein support",
        "Mean Platelet Volume": "B₁₂ & folate; vitamin B₆",
    }

    # Suggestions for elevations
    HIGH_SUGGESTIONS: ClassVar[Dict[str, str]] = {
        "Hemoglobin": "avoid excess iron supplements; stay well-hydrated; moderate red meat intake",
        "Packed Cell Volume (PCV)": "maintain hydration; limit high-salt or high-protein loads",
        "RBC Count": "hydration and balanced diet to prevent hemoconcentration",
        "MCV": "boost B₁₂ & folate (eggs, liver, avocado); avoid excessive alcohol",
        "MCH": "ensure balanced B₁₂ & folate; moderate iron intake",
        "MCHC": "maintain hydration; balanced intake of healthy fats (olive oil, nuts)",
        "RDW": "focus on balanced micronutrients; multivitamins as needed",
        "Total Leukocyte Count (TLC)": "anti-inflammatory diet: omega-3 (fatty fish, chia); antioxidants (berries, green tea)",
        "Segmented Neutrophils": "support with probiotics (yogurt, kefir); antioxidants (vitamin C fruits)",
        "Lymphocytes": "reduce chronic inflammation: turmeric, ginger; green leafy vegetables",
        "Monocytes": "anti-inflammatory foods: omega-3 (flaxseed, salmon); garlic; cumin",
        "Eosinophils": "allergy-friendly diet: avoid histamine-rich foods (aged cheese, fermented); increase quercetin (apples)",
        "Basophils": "low-histamine diet (fresh produce); vitamin C to stabilize mast cells",
        "Platelet Count": "anti-inflammatory foods: berries, olive oil; reduce refined carbohydrates",
        "Mean Platelet Volume": "balanced healthy fats (avocado, nuts); stay hydrated",
    }

    def _run(self, summary_text: str, weight_kg: float) -> str:
        """
        summary_text: the text output from blood_report_summary
        weight_kg: patient's weight in kilograms
        Returns advice only for tests flagged 'deficiency' or 'elevation' and protein requirement.
        """
        advice_lines: list[str] = []
        # Protein: 1.2 g per kg
        protein_req = round(weight_kg * 1.2)
        advice_lines.append(
            f"• Protein Intake: aim for approximately {protein_req} g per day from eggs, dairy, legumes, and lean meats."
        )

        # Only parse abnormal lines (deficiency or elevation)
        for line in summary_text.splitlines():
            match = re.match(r"- ([^:]+): .*→\s*(deficiency|elevation)", line, re.IGNORECASE)
            if not match:
                continue

            test_name = match.group(1).strip()
            status_key = match.group(2).lower()

            if status_key == "deficiency":
                suggestion = self.LOW_SUGGESTIONS.get(test_name)
                if suggestion:
                    advice_lines.append(f"• {test_name} deficiency: Increase intake of {suggestion}.")
                else:
                    advice_lines.append(f"• {test_name} deficiency: Consult your doctor for dietary guidance.")

            elif status_key == "elevation":
                suggestion = self.HIGH_SUGGESTIONS.get(test_name)
                if suggestion:
                    advice_lines.append(f"• {test_name} elevation: Recommended dietary changes include {suggestion}.")
                else:
                    advice_lines.append(f"• {test_name} elevation: Consult your doctor for dietary guidance.")

        return "\n".join(advice_lines)




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
        if api_key:
            headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
            # Fetch top 5 websites for yoga practice
            try:
                resp = requests.post(
                    "https://api.serper.dev/search",
                    headers=headers,
                    json={"q": "best yoga practice websites", "num": 5},
                    timeout=5,
                )
                resp.raise_for_status()
                data = resp.json()
                items = data.get("organic") or data.get("results") or []
                suggestions.append("\nTop 5 Yoga Practice Websites:")
                for item in items[:5]:
                    link = item.get("link") or item.get("url")
                    suggestions.append(f"  - {link}")
            except Exception as e:
                logger.error(f"Error fetching yoga websites: {e}")

            # Fetch top 5 YouTube videos for yoga practice
            try:
                resp = requests.post(
                    "https://api.serper.dev/search",
                    headers=headers,
                    json={"q": "best yoga practice YouTube videos", "num": 5},
                    timeout=5,
                )
                resp.raise_for_status()
                data = resp.json()
                items = data.get("organic") or data.get("results") or []
                suggestions.append("\nTop 5 Yoga Practice YouTube Videos:")
                for item in items[:5]:
                    link = item.get("link") or item.get("url")
                    suggestions.append(f"  - {link}")
            except Exception as e:
                logger.error(f"Error fetching yoga videos: {e}")
        else:
            suggestions.append("\n• SERPER_API_KEY not set; cannot fetch external links.")

        return "\n".join(suggestions)



# EXPORTED SYMBOLS
# Declares all items that should be imported when doing:
#     from tools import *
__all__ = [
    "BloodTestReportTool",
    "AbnormalInfoSearchTool",
    "BloodReportSummaryTool",
    "NutritionTool",
    "ExerciseTool",
    "search_tool",
]

# Quick test block
if __name__ == "__main__":
    # Quick test for the search tool
    search_tool = SerperDevTool()
    print(search_tool("What is anemia?"))

    # Quick test of BloodTestReportTool (uncomment if you have a test file)
    # tool = BloodTestReportTool()
    # result = tool._run("data/sample.pdf")
    # print(result)