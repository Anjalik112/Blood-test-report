import motor.motor_asyncio
import os

# Load MongoDB connection string from environment variables
# This helps keep credentials secure instead of hard-coding them
MONGO_URI = os.getenv("MONGO_URI")

# Name of the database to use
DB_NAME = "blood_report_test_db"

# CHANGED:
# Create an asynchronous MongoDB client using Motor.
# Motor integrates seamlessly with async frameworks like FastAPI.
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)

# Access the specific database
db = client[DB_NAME]

# Get the collection to store blood report analyses
# Each document will store:
# - user_name
# - query
# - analysis result
# - original file name
# - timestamp
reports_collection = db["reports"]
