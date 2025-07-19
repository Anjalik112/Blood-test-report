from celery import Celery
import os

# Load environment variables from .env file
# This ensures we can configure Celery using variables like:
# - CELERY_BROKER_URL
# - CELERY_RESULT_BACKEND
from dotenv import load_dotenv
load_dotenv()

# Create the Celery application instance
# This acts as the entry point for defining and running asynchronous tasks
celery_app = Celery(
    'tasks',  # Name of the app (can be any string)
    broker=os.getenv(
        'CELERY_BROKER_URL',
        'redis://localhost:6379/0'  # Default to Redis if not set in .env
    )
)

# Optional: Configure where task results will be stored
# This enables you to query task results after execution
celery_app.conf.result_backend = os.getenv(
    'CELERY_RESULT_BACKEND',
    'redis://localhost:6379/0'
)

# Optional: Set a time limit for how long results are kept
# Helps avoid growing the backend indefinitely
celery_app.conf.result_expires = 3600  # 1 hour expiration time
