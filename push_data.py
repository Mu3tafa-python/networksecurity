
import os
import sys
from pathlib import Path

import pandas as pd
import pymongo
import certifi
from dotenv import load_dotenv

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


# Load environment variables from .env
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")


class NetworkDataExtract:

    def __init__(self):
        try:
            if not MONGO_DB_URL:
                raise ValueError(
                    "MONGO_DB_URL is missing from the .env file."
                )

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def csv_to_json_convertor(self, file_path):
        try:
            data = pd.read_csv(file_path)

            # Convert CSV records into a list of dictionaries
            records = data.to_dict(orient="records")

            logging.info(
                f"CSV converted successfully. Records: {len(records)}"
            )

            return records

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def insert_data_mongodb(self, records, database, collection):
        try:
            if not records:
                raise ValueError("No records found to insert.")

            # Connect to MongoDB Atlas
            with pymongo.MongoClient(
                MONGO_DB_URL,
                tlsCAFile=certifi.where(),
                serverSelectionTimeoutMS=10000
            ) as mongo_client:

                # Verify the database connection
                mongo_client.admin.command("ping")
                print("Successfully connected to MongoDB Atlas!")

                # Select database and collection
                db = mongo_client[database]
                data_collection = db[collection]

                # Insert records into MongoDB
                result = data_collection.insert_many(records)

                inserted_count = len(result.inserted_ids)

                logging.info(
                    f"Successfully inserted {inserted_count} records."
                )

                return inserted_count

        except Exception as e:
            raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    try:
        # Update these paths to match your actual folder and CSV filename
        FILE_PATH = Path("Network_Data") / "phisingData.csv"

        DATABASE = "MUSTAFA-AI"
        COLLECTION = "NetworkData"

        if not FILE_PATH.exists():
            raise FileNotFoundError(
                f"CSV file not found: {FILE_PATH.resolve()}"
            )

        network_obj = NetworkDataExtract()

        records = network_obj.csv_to_json_convertor(
            file_path=FILE_PATH
        )

        print(f"Total records read from CSV: {len(records)}")

        inserted_count = network_obj.insert_data_mongodb(
            records=records,
            database=DATABASE,
            collection=COLLECTION
        )

        print(f"Total records inserted: {inserted_count}")

    except Exception as e:
        print(f"Data upload failed: {e}")
