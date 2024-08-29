import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pymongo import MongoClient
import matplotlib.pyplot as plt

# Define the MongoDB connection details
mongo_uri = "mongodb://localhost:27017/"
database_name = "mydatabase"
collection_name = "mycollection"

# Connect to MongoDB
client = MongoClient(mongo_uri)
db = client[database_name]
collection = db[collection_name]

# Define the Excel file path and sheet names
file_path = 'C:/Users/OMEN/StockMarket/Correlation_Tables.xlsx'  # Adjust file path accordingly!
sheet_names = ['12mo', '9mo', '6mo', '3mo']

# Create an empty dictionary to store the Series
result = {}

# Read and process each sheet
for sheet in sheet_names:
    df = pd.read_excel(file_path, sheet_name=sheet, usecols='B:C', skiprows=1, nrows=10)
    series = df.apply(lambda row: ', '.join(row.astype(str)), axis=1)
    result[sheet] = series
    print(f"Series from {sheet} (B3:C12):")
    print(series)
    print("\n")

# Union the Series from the different sheets
sets = [set(series) for series in result.values()]
union_set = sets[0].union(*sets[1:])
selected_features = list(union_set)
print("Features (union of all Series):")
print(selected_features)

# Clean and format selected features
selected_features = [feature.split(',')[0].strip() for feature in selected_features]

# Read data from MongoDB collection and define features and targets
cursor = collection.find({}, {feature: 1 for feature in selected_features + ['close']})
data = pd.DataFrame(list(cursor))

# Remove the MongoDB ObjectId column if it exists
if '_id' in data.columns:
    data.drop('_id', axis=1, inplace=True)

# Convert numeric columns to proper data types, ignoring errors
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data.fillna(method='ffill', inplace=True)
data.dropna(axis=1, how='any', inplace=True)

# Check if all selected features exist in the data
missing_features = [feature for feature in selected_features if feature not in data.columns]
if missing_features:
    print(f"Missing features in data: {missing_features}")
    selected_features = [feature for feature in selected_features if feature in data.columns]

# Proceed if there are any selected features left
if selected_features:
    features = data[selected_features]

    # Create target columns
    target_cols = {
        'close_3mo': data['close'].shift(-90),
        'close_6mo': data['close'].shift(-180),
        'close_9mo': data['close'].shift(-270),
        'close_12mo': data['close'].shift(-365)
    }

    # Add target columns to the DataFrame using pd.concat
    data = pd.concat([data, pd.DataFrame(target_cols)], axis=1)

    # Drop rows with NaN values in target columns and features
    data.dropna(subset=['close_3mo', 'close_6mo', 'close_9mo', 'close_12mo'] + selected_features, inplace=True)

    # Redefine features and targets after dropping NaNs
    features = data[selected_features]
    target_3mo = data['close_3mo']
    target_6mo = data['close_6mo']
    target_9mo = data['close_9mo']
    target_12mo = data['close_12mo']

    # Debugging: Check first few rows of targets
    print("First few rows of targets:")
    print(data[['close', 'close_3mo', 'close_6mo', 'close_9mo', 'close_12mo']].head())

    # Scale the target variables
    scaler_y = StandardScaler()
    target_3mo = scaler_y.fit_transform(target_3mo.values.reshape(-1, 1)).flatten()
    target_6mo = scaler_y.fit_transform(target_6mo.values.reshape(-1, 1)).flatten()
    target_9mo = scaler_y.fit_transform(target_9mo.values.reshape(-1, 1)).flatten()
    target_12mo = scaler_y.fit_transform(target_12mo.values.reshape(-1, 1)).flatten()

    # Standardize data
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    print("Features scaled successfully.")
    
    # Ensure all lengths are consistent
    assert len(features_scaled) == len(target_3mo) == len(target_6mo) == len(target_9mo) == len(target_12mo), "Inconsistent lengths found!"

    # Verify the shapes of the training and testing sets
    X_train_3mo, X_test_3mo, y_train_3mo, y_test_3mo = train_test_split(features_scaled, target_3mo, test_size=0.2, random_state=42)
    X_train_6mo, X_test_6mo, y_train_6mo, y_test_6mo = train_test_split(features_scaled, target_6mo, test_size=0.2, random_state=42)
    X_train_9mo, X_test_9mo, y_train_9mo, y_test_9mo = train_test_split(features_scaled, target_9mo, test_size=0.2, random_state=42)
    X_train_12mo, X_test_12mo, y_train_12mo, y_test_12mo = train_test_split(features_scaled, target_12mo, test_size=0.2, random_state=42)
    
    print("Shapes after train-test split:")
    print("X_train_3mo:", X_train_3mo.shape)
    print("X_test_3mo:", X_test_3mo.shape)
    print("y_train_3mo:", y_train_3mo.shape)
    print("y_test_3mo:", y_test_3mo.shape)
    print("X_train_6mo:", X_train_6mo.shape)
    print("X_test_6mo:", X_test_6mo.shape)
    print("y_train_6mo:", y_train_6mo.shape)
    print("y_test_6mo:", y_test_6mo.shape)
    print("X_train_9mo:", X_train_9mo.shape)
    print("X_test_9mo:", X_test_9mo.shape)
    print("y_train_9mo:", y_train_9mo.shape)
    print("y_test_9mo:", y_test_9mo.shape)
    print("X_train_12mo:", X_train_12mo.shape)
    print("X_test_12mo:", X_test_12mo.shape)
    print("y_train_12mo:", y_train_12mo.shape)
    print("y_test_12mo:", y_test_12mo.shape)
else:
    print("No valid features found in the data.")

# Define a neural network model using TensorFlow with dropout regularization
initial_learning_rate = 0.0001

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train_3mo.shape[1],)),  # Reverting complexity
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)  # Standard optimizer
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

model_3mo = build_model()
model_6mo = build_model()
model_9mo = build_model()
model_12mo = build_model()

# Callbacks for early stopping and learning rate reduction
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Custom callback for data verification
class DataVerificationCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"Epoch {epoch+1} started with data shapes:")
        print(f"Features: {X_train_3mo.shape}")
        print(f"Target 3mo: {y_train_3mo.shape}")
        print(f"Target 6mo: {y_train_6mo.shape}")
        print(f"Target 9mo: {y_train_9mo.shape}")
        print(f"Target 12mo: {y_train_12mo.shape}")

# Train the model based on training data with callbacks and batch size of 32
batch_size = 32  # Decreased batch size

history_3mo = model_3mo.fit(X_train_3mo, y_train_3mo, epochs=200, validation_split=0.2, batch_size=batch_size, callbacks=[early_stopping, lr_scheduler, DataVerificationCallback()])
history_6mo = model_6mo.fit(X_train_6mo, y_train_6mo, epochs=200, validation_split=0.2, batch_size=batch_size, callbacks=[early_stopping, lr_scheduler, DataVerificationCallback()])
history_9mo = model_9mo.fit(X_train_9mo, y_train_9mo, epochs=200, validation_split=0.2, batch_size=batch_size, callbacks=[early_stopping, lr_scheduler, DataVerificationCallback()])
history_12mo = model_12mo.fit(X_train_12mo, y_train_12mo, epochs=200, validation_split=0.2, batch_size=batch_size, callbacks=[early_stopping, lr_scheduler, DataVerificationCallback()])

# Evaluate model and print loss
loss_3mo = model_3mo.evaluate(X_test_3mo, y_test_3mo)
loss_6mo = model_6mo.evaluate(X_test_6mo, y_test_6mo)
loss_9mo = model_9mo.evaluate(X_test_9mo, y_test_9mo)
loss_12mo = model_12mo.evaluate(X_test_12mo, y_test_12mo)

print(f'3 Months Prediction Loss: {loss_3mo}')
print(f'6 Months Prediction Loss: {loss_6mo}')
print(f'9 Months Prediction Loss: {loss_9mo}')
print(f'12 Months Prediction Loss: {loss_12mo}')

# Predict future closing prices
predictions_3mo = model_3mo.predict(X_test_3mo)
predictions_6mo = model_6mo.predict(X_test_6mo)
predictions_9mo = model_9mo.predict(X_test_9mo)
predictions_12mo = model_12mo.predict(X_test_12mo)

# Visualise predictions and compare with actual values
def plot_predictions(actual, predicted, title):
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.title(title)
    plt.legend()
    plt.show()

plot_predictions(y_test_3mo, predictions_3mo, '3 Months Prediction')
plot_predictions(y_test_6mo, predictions_6mo, '6 Months Prediction')
plot_predictions(y_test_9mo, predictions_9mo, '9 Months Prediction')
plot_predictions(y_test_12mo, predictions_12mo, '12 Months Prediction')