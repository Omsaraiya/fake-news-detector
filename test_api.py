import requests

url = 'http://127.0.0.1:5000/predict'

data = {
    "text": "Miracle cure discovered! Drink this everyday to live forever and become a billionaire."
}

print("Sending data to the AI Server...")
response = requests.post(url, json=data)

print("\n--- AI RESPONSE ---")
print(response.json())