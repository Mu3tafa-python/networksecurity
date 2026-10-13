
import os

import certifi
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

uri = os.getenv("MONGO_DB_URL")

if not uri:
    raise ValueError("MONGO_DB_URL is missing from .env")

try:
    with MongoClient(
        uri,
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=10000
    ) as client:

        client.admin.command("ping")
        print("Successfully connected to MongoDB Atlas!")

except Exception as e:
    print(f"MongoDB connection failed: {e}")
