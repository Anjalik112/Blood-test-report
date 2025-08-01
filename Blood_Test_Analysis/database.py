import motor.motor_asyncio
import os

MONGO_URI = os.getenv("MONGO_URI")

# Name of the database to use
DB_NAME = "blood_report_test_db"

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)

# Access the specific database
db = client[DB_NAME]

reports_collection = db["reports"]
