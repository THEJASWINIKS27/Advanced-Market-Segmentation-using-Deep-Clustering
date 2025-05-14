import pandas as pd
import joblib
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load raw data
data = pd.read_csv("customer_orders.csv")

# Preprocessing as in notebook
data['shipping_date'] = pd.to_datetime(data['shipping_date'], errors='coerce', utc=True)
data['shipment_duration'] = (pd.Timestamp.now(tz='UTC') - data['shipping_date']).dt.days
median_duration = data['shipment_duration'].median()
data['shipment_duration'] = data['shipment_duration'].apply(lambda x: x if 0 <= x <= 120 else median_duration)
data.ffill(inplace=True)
data['total_sales_per_customer'] = data.groupby('customer_id')['sales_per_customer'].transform('sum')
data['orders_per_customer'] = data.groupby('customer_id')['order_id'].transform('count')
data = data.drop(columns=['shipping_date', 'order_date', 'product_name', 'category_name'], errors='ignore')

# Define features
numerical_features = ['profit_per_order', 'sales_per_customer', 'total_sales_per_customer', 'orders_per_customer', 'shipment_duration']
categorical_features = ['payment_type', 'customer_country', 'customer_segment']

# Load preprocessor
preprocessor = joblib.load("preprocessor.pkl")

# Preprocess data
data_processed = preprocessor.transform(data)

# Load encoder model
encoder = tf.keras.models.load_model("encoder_model.h5")

# Encode data
encoded_data = encoder.predict(data_processed)

# Train KMeans on encoded data
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(encoded_data)

# Save new KMeans model
joblib.dump(kmeans, "kmeans_model.pkl")

print("KMeans model retrained and saved successfully.")
