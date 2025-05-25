import requests

files = {'file': open('sample_test_data.csv', 'rb')}
try:
    response = requests.post("http://127.0.0.1:8000/predict/", files=files)
    response.raise_for_status()
    print("Prediction response:")
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
