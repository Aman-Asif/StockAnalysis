from pymongo import MongoClient

# Connect to your MongoDB database
client = MongoClient("mongodb://localhost:27017/")
db = client.mydatabase

def update_main_collection(period_days, input_collection_name):
    input_collection = db[input_collection_name]
    suffix = f"_{period_days}d"  # Suffix to differentiate periods

    pipeline = [
        {"$sort": {"symbol": 1, "date": 1}},  # Sort by symbol and date
        {"$group": {  # Group by symbol and create arrays for date and close
            "_id": "$symbol",
            "dates": {"$push": "$date"},
            "closes": {"$push": "$close"},
            "docs": {"$push": {"_id": "$_id", "date": "$date", "close": "$close"}}
        }},
        {"$unwind": {"path": "$docs", "includeArrayIndex": "arrayIndex"}},  # Unwind the documents array
        {"$set": {  # Calculate newdate and future elements for close_diff
            "symbol": "$_id",
            "date": "$docs.date",
            "_id": "$docs._id",
            "close": "$docs.close",
            f"newdate{suffix}": {"$dateAdd": {"startDate": "$docs.date", "unit": "day", "amount": period_days}},
            f"nextClose{suffix}": {"$arrayElemAt": ["$closes", {"$add": ["$arrayIndex", period_days]}]},
            f"close_diff{suffix}": {"$subtract": [{"$arrayElemAt": ["$closes", {"$add": ["$arrayIndex", period_days]}]}, "$docs.close"]}
        }},
        {"$project": {  # Project only the necessary fields for merge
            "_id": 1,
            f"newdate{suffix}": 1,
            f"nextClose{suffix}": 1,
            f"close_diff{suffix}": 1,
        }},
        {"$merge": {  # Merge the results back into the main collection
            "into": input_collection_name,
            "whenMatched": "merge",
            "whenNotMatched": "discard"
        }}
    ]

    # Execute the aggregation pipeline
    input_collection.aggregate(pipeline, allowDiskUse=True)

    # Print confirmation message
    print(f"The main collection has been updated with the 'newdate{suffix}', 'nextClose{suffix}', and 'close_diff{suffix}' fields for the {period_days}-day period.")

def create_output_collection(period_days, input_collection_name, output_collection_name):
    input_collection = db[input_collection_name]
    suffix = f"_{period_days}d"  # Suffix to differentiate periods

    pipeline = [
        {"$sort": {"symbol": 1, "date": 1}},  # Sort by symbol and date
        {"$group": {  # Group by symbol and create arrays for date and close
            "_id": "$symbol",
            "dates": {"$push": "$date"},
            "closes": {"$push": "$close"},
            "docs": {"$push": {"_id": "$_id", "date": "$date", "close": "$close"}}
        }},
        {"$unwind": {"path": "$docs", "includeArrayIndex": "arrayIndex"}},  # Unwind the documents array
        {"$set": {  # Calculate newdate and future elements for close_diff
            "symbol": "$_id",
            "date": "$docs.date",
            "_id": "$docs._id",
            "close": "$docs.close",
            f"newdate{suffix}": {"$dateAdd": {"startDate": "$docs.date", "unit": "day", "amount": period_days}},
            f"nextClose{suffix}": {"$arrayElemAt": ["$closes", {"$add": ["$arrayIndex", period_days]}]},
            f"close_diff{suffix}": {"$subtract": [{"$arrayElemAt": ["$closes", {"$add": ["$arrayIndex", period_days]}]}, "$docs.close"]}
        }},
        {"$project": {  # Project only the necessary fields
            "symbol": 1,
            "close": 1,
            "_id": 1,
            "date": 1,
            f"newdate{suffix}": 1,
            f"close_diff{suffix}": 1,
        }},
        {"$out": output_collection_name}  # Output to the new collection
    ]

    # Execute the aggregation pipeline
    input_collection.aggregate(pipeline, allowDiskUse=True)

    # Print confirmation message
    print(f"The new collection '{output_collection_name}' for the {period_days}-day period has been created with the 'newdate{suffix}' and 'close_diff{suffix}' fields.")

# Define the input collection name
input_collection_name = "mycollection"

# Update the main collection and create output collections for different periods
update_main_collection(365, input_collection_name)
create_output_collection(365, input_collection_name, "regressioncollection12mo")

update_main_collection(270, input_collection_name)
create_output_collection(270, input_collection_name, "regressioncollection9mo")

update_main_collection(180, input_collection_name)
create_output_collection(180, input_collection_name, "regressioncollection6mo")

update_main_collection(90, input_collection_name)
create_output_collection(90, input_collection_name, "regressioncollection3mo")