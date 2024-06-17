import requests

url = "http://localhost:5000/predict"
data = {
    'taxi_type': 'yellow',
    'year': 2023,
    'month': 5
}

response = requests.post(url, json=data)
print(response.json())
