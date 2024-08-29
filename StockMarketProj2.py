import numpy as np
from scipy.stats import pearsonr
from pymongo import MongoClient

def calculate_correlation(x, y):
    """Calculate Pearson correlation coefficient between two lists, removing infs and NaNs."""
    x = np.array(x)
    y = np.array(y)
    mask = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isinf(x)) & (~np.isinf(y)) & (x != None) & (y != None)
    if np.sum(mask) == 0:  # Check if all values are filtered out
        return None
    x = x[mask]
    y = y[mask]
    if np.all(x == x[0]) or np.all(y == y[0]):  # Check if input arrays are constant
        return None
    return pearsonr(x, y)[0]

def fetch_batch(collection, regression_collection, skip, limit):
    pipeline = [
        {"$skip": skip},
        {"$limit": limit},
        {
            "$lookup": {
                "from": regression_collection,
                "localField": "_id",
                "foreignField": "_id",
                "as": "joined_docs"
            }
        },
        {"$unwind": "$joined_docs"},
        {"$set": {"close_diff": "$joined_docs.close_diff"}},
        {"$match": {"close_diff": {"$exists": True}}},  # Filter out documents where close_diff does not exist
        {"$project": {"_id": 0, "close_diff": 1, "fields": "$$ROOT"}}
    ]
    return list(collection.aggregate(pipeline, allowDiskUse=True))

def calculate_correlations(input_collection_name, regression_collection_name, batch_size=100000):
    # Connection setup
    print("Setting up MongoDB connection...")
    connection_string = "mongodb://localhost:27017/"
    client = MongoClient(connection_string)
    db = client['mydatabase']
    input_collection = db[input_collection_name]
    regression_collection = db[regression_collection_name]

    # Initialize variables
    skip = 0
    total_docs = input_collection.count_documents({})
    close_diff_values = []
    field_values = {}

    # Fetch and process data in batches
    while skip < total_docs:
        print(f"Fetching batch starting from {skip}...")
        batch_data = fetch_batch(input_collection, regression_collection_name, skip, batch_size)
        print(f"Fetched {len(batch_data)} documents in this batch.")
        
        for doc in batch_data:
            close_diff = doc.get('close_diff')
            if close_diff is not None:
                close_diff_values.append(close_diff)
                for field, value in doc['fields'].items():
                    if isinstance(value, (int, float)):
                        if value is not None:
                            if field not in field_values:
                                field_values[field] = []
                            field_values[field].append(value)
        skip += batch_size

    print(f"Total close_diff values collected: {len(close_diff_values)}")
    print(f"Total fields collected: {len(field_values)}")
    print("Field values collected for each field:")
    for field_name, values in field_values.items():
        print(f"{field_name}: {len(values)} values")

    # Calculate correlations
    print("Calculating correlations...")
    correlations = {}
    for field_name, values in field_values.items():
        if field_name == "close_diff":  # Skip close_diff field to avoid self-correlation
            continue
        min_length = min(len(close_diff_values), len(values))
        correlation = calculate_correlation(close_diff_values[:min_length], values[:min_length])
        if correlation is not None and not np.isnan(correlation):
            correlations[field_name] = correlation

    # Sort the correlations
    print("Sorting correlations...")
    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    # Print all correlations
    print(f"Correlations for each field with close_diff (ordered from largest to smallest) over period from {regression_collection_name}:")
    for col, corr_value in sorted_correlations:
        print(f"{col}: {corr_value}")
    print("Process completed.")

calculate_correlations('mycollection', 'regressioncollection12mo')
calculate_correlations('mycollection', 'regressioncollection9mo')
calculate_correlations('mycollection', 'regressioncollection6mo')
calculate_correlations('mycollection', 'regressioncollection3mo')