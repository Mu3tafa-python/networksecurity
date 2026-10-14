
from dotenv import load_dotenv
import os
import pymongo
import dns.resolver

load_dotenv()

mongo_url = os.getenv("MONGO_DB_URL")

if not mongo_url:
    raise ValueError("MONGO_DB_URL is missing from .env")

# Use public DNS servers to resolve MongoDB SRV records.
resolver = dns.resolver.Resolver()
resolver.nameservers = ["8.8.8.8", "1.1.1.1"]
dns.resolver.default_resolver = resolver

client = pymongo.MongoClient(
    mongo_url,
    serverSelectionTimeoutMS=20000,
    connectTimeoutMS=15000,
)

try:
    client.admin.command("ping")
    print("MongoDB ping: SUCCESS")

    collection = client["MUSTAFA-AI"]["NetworkData"]

    print("Total documents:", collection.count_documents({}))
    print("Result distribution:")

    pipeline = [
        {"$group": {"_id": "$Result", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
    ]

    for item in collection.aggregate(pipeline):
        print(item)

finally:
    client.close()
