from flask import Flask, request, jsonify
import pandas as pd
from sklearn.cluster import KMeans
from io import StringIO
import os
import joblib
import tensorflow as tf

app = Flask(__name__)

# Load pre-trained KMeans model
model_path = "kmeans_model.pkl"
kmeans = joblib.load(model_path)

# Load preprocessing pipeline and encoder model
preprocessor = joblib.load("preprocessor.pkl")
encoder = tf.keras.models.load_model("encoder_model.h5")

# Required raw input columns for preprocessing
required_columns = [
    'profit_per_order', 'sales_per_customer',
    'payment_type',
    'customer_country', 'customer_segment', 'order_id', 'customer_id',
    'shipping_date'
]

@app.route("/")
def home():
    return "Customer Segmentation Backend is running."

@app.route("/predict/", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    content = file.stream.read().decode("utf-8")  # Read file contents
    df = pd.read_csv(StringIO(content))  # Convert to DataFrame

    # Check for required columns
    important_columns = [
        'category_id', 'customer_city', 'customer_country', 'customer_id',
        'customer_segment', 'customer_state', 'customer_zipcode', 'department_id',
        'department_name', 'label', 'latitude', 'longitude', 'market'
    ]

    missing_important_cols = [col for col in important_columns if col not in df.columns]
    if missing_important_cols:
        return jsonify({"error": f"Missing important columns for display: {', '.join(missing_important_cols)}"}), 400

    for col in required_columns:
        if col not in df.columns:
            return jsonify({"error": f"Missing required column: {col}"}), 400

    # Feature engineering as in training
    df['shipping_date'] = pd.to_datetime(df['shipping_date'], errors='coerce', utc=True)
    df['shipment_duration'] = (pd.Timestamp.now(tz='UTC') - df['shipping_date']).dt.days
    median_duration = df['shipment_duration'].median()
    df['shipment_duration'] = df['shipment_duration'].apply(lambda x: x if 0 <= x <= 120 else median_duration)
    df.ffill(inplace=True)
    df['total_sales_per_customer'] = df.groupby('customer_id')['sales_per_customer'].transform('sum')
    df['orders_per_customer'] = df.groupby('customer_id')['order_id'].transform('count')
    df = df.drop(columns=['shipping_date', 'order_date', 'product_name', 'category_name'], errors='ignore')

    # Preprocess and encode
    processed_data = preprocessor.transform(df)
    encoded_data = encoder.predict(processed_data)

    # Predict segments
    print(f"Encoded data shape: {encoded_data.shape}")
    print(f"KMeans expected input shape: {kmeans.cluster_centers_.shape}")
    if encoded_data.shape[1] != kmeans.cluster_centers_.shape[1]:
        error_msg = (f"Feature mismatch: encoded data has {encoded_data.shape[1]} features, "
                     f"but KMeans expects {kmeans.cluster_centers_.shape[1]} features.")
        print(error_msg)
        return jsonify({"error": error_msg}), 500
    segments = kmeans.predict(encoded_data)
    df["Segment"] = segments

    return jsonify(df.to_dict(orient="records"))

if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 8000))
        # Change host to 0.0.0.0 to allow connections from all interfaces
        app.run(host="0.0.0.0", port=port, debug=True)
    except Exception as e:
        import traceback
        print("Failed to start backend server:")
        traceback.print_exc()
