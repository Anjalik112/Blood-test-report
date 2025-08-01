import streamlit as st
import requests

# FastAPI endpoint URL
api_url = "http://127.0.0.1:8000/analyze"

st.title("Blood Test Report Analyzer")
st.subheader("Upload a PDF and get segmented results from each agent")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
query = st.text_area("Enter your query", placeholder="Write your query here", height=100)

if st.button("Analyze Report"):
    if not uploaded_file:
        st.warning("⚠️ Please upload a PDF file first.")
        st.stop()
    if not query.strip():
        st.warning("⚠️ Please enter a query.")
        st.stop()

    st.write(f"Uploaded file: **{uploaded_file.name}**")
    files = {"file": (uploaded_file.name, uploaded_file.read(), "application/pdf")}
    data  = {"query": query}

    try:
        resp = requests.post(api_url, files=files, data=data)
        resp.raise_for_status()
    except Exception as e:
        st.error(f"API request failed: {e}")
        st.stop()

    result = resp.json()
    st.success("✅ Analysis Completed!")

    # Metadata
    st.markdown(f"**User Name:** {result.get('user_name','N/A')}")
    st.markdown(f"**Query:** {result.get('query','N/A')}")
    st.markdown(f"**File:** {result.get('file_processed','N/A')}")
    st.markdown(f"**Report ID:** `{result.get('report_id','N/A')}`")

    # Now show each agent’s output
    st.subheader("Results by Agent")
    sections = [
        ("Doctor’s Analysis",   "doctor_report"),
        ("Abnormality Info",     "abnormal_info"),
        ("Nutrition Plan",       "nutrition_plan"),
        ("Exercise Routine",     "exercise_routine"),
    ]
    for title, key in sections:
        content = result.get(key, "")
        if content.startswith("Error:"):
            st.error(f"**{title}**\n\n{content}")
        else:
            st.markdown(f"**{title}**")
            st.write(content or "No output returned.")

