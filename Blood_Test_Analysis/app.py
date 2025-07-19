import streamlit as st
import requests

# FastAPI endpoint URL
# Update this if your API is running on a different host or port.
api_url = "http://127.0.0.1:8000/analyze"

# ==============================
# STREAMLIT FRONTEND
# ==============================

# Set page title
st.title("Blood Test Report Analyzer")

# Subheader for instructions
st.subheader("Upload a Blood Test Report PDF and get the analysis")

# File uploader widget
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# Query input field
# Default query provides a helpful prompt for new users.
query = st.text_area(
    "Enter your query",
    placeholder="Write your query here",
    height=100
)

# Analyze button triggers the request
if st.button("Analyze Report"):
    if uploaded_file is None:
        # User did not upload a PDF
        st.warning("⚠️ Please upload a PDF file first.")
    elif not query.strip():
        # User left the query empty
        st.warning("⚠️ Please enter a query.")
    else:
        st.write(f"Uploaded file: {uploaded_file.name}")

        # Read uploaded file bytes to send in HTTP POST request
        file_bytes = uploaded_file.read()

        # Prepare multipart/form-data payload
        files = {
            "file": (uploaded_file.name, file_bytes, "application/pdf")
        }
        data = {
            "query": query
        }

        # Send request to FastAPI backend
        try:
            response = requests.post(api_url, files=files, data=data)
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
            st.stop()

        # Handle successful response
        if response.status_code == 200:
            result = response.json()

            st.success("✅ Analysis Completed!")

            # Display details from the API response
            st.markdown(f"**User Name:** {result.get('user_name', 'N/A')}")
            st.markdown(f"**Query:** {result.get('query', 'N/A')}")
            st.markdown(f"**Original File Name:** {result.get('file_processed', 'N/A')}")
            st.markdown(f"**Timestamp:** {result.get('timestamp', 'N/A')}")
            st.markdown(f"**Report ID:** `{result.get('report_id', 'N/A')}`")

            st.subheader("Analysis Text:")
            st.write(result.get('analysis', 'No analysis returned.'))

        else:
            # Show backend error details
            st.error(f"❌ Error from FastAPI: {response.text}")
