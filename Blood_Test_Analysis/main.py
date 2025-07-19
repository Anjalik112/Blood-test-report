from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import os
import uuid
from datetime import datetime
import logging
import pdfplumber

# CHANGED: switched from original blocking crew logic to a modular pipeline
from crew_runner import run_crew_pipeline  

# NEW: integrated MongoDB for storing analysis results
from database import reports_collection  # MongoDB collection (ensure it's async)

app = FastAPI(title="Blood Test Report Analyser")

# NEW: added logging for better tracing and debugging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def extract_user_name_from_pdf(file_path: str) -> str:
    """
    Simple function to extract 'Name' from the PDF text.
    This can be improved for more complex formats.
    """
    user_name = None

    try:
        with pdfplumber.open(file_path) as pdf:
            text = ''
            for page in pdf.pages:
                text += page.extract_text() or ''

        for line in text.splitlines():
            if 'Name' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    user_name = parts[1].strip()
                    break
    except Exception as e:
        logger.warning(f"Failed to extract name from PDF: {e}")

    return user_name or 'Unknown User'

@app.post("/analyze")
async def analyze_blood_report(
    file: UploadFile = File(...),
    query: str      = Form(...),
) -> JSONResponse:
    # 1) Save upload and extract name
    file_id = str(uuid.uuid4())
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, f'blood_test_report_{file_id}.pdf')

    content = await file.read()
    with open(file_path, 'wb') as f:
        f.write(content)

    user_name = extract_user_name_from_pdf(file_path)

    # 2) Run pipeline (returns filtered sections)
    try:
        reports = run_crew_pipeline(query=query.strip(), file_path=file_path)
    except Exception as e:
        logger.exception("Error during report analysis.")
        raise HTTPException(status_code=502, detail=f"Pipeline error: {e}")
    finally:
        # Cleanup temp file
        try:
            os.remove(file_path)
        except Exception:
            pass

    # 3) Build document for MongoDB
    document = {
        'user_name':            user_name,
        'query':                query,
        **reports,  # includes doctor_report, abnormal_info, etc.
        'original_file_name':   file.filename,
        'timestamp':            datetime.utcnow(),
        'db_error':             None,
        'report_id':            None,
    }

    # 4) Insert into MongoDB
    try:
        result = await reports_collection.insert_one(document)
        document['report_id'] = str(result.inserted_id)
    except Exception as db_err:
        logger.exception("Error saving to MongoDB.")
        document['db_error'] = str(db_err)

    # 5) Construct response
    response = {
        'status':         'success',
        'user_name':      document['user_name'],
        'query':          document['query'],
        **{k: document[k] for k in reports.keys()},
        'file_processed': document['original_file_name'],
        'report_id':      document['report_id'],
    }
    if document.get('db_error'):
        response['db_error'] = document['db_error']

    return JSONResponse(content=response)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)
