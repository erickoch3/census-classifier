import requests
import json

# URL of the prediction endpoint
url = "http://census-classifier-env.eba-t23zcbut.us-east-1.elasticbeanstalk.com/predict"

# Data for the first POST request
data1 = {
    "age": 25,
    "workclass": "Private",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

# Data for the second POST request
data2 = {
    "age": 29,
    "workclass": "Private",
    "fnlgt": 185908,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "Black",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 55,
    "native-country": "United-States"
}

# Headers for the POST requests
headers = {
    "Content-Type": "application/json"
}

# Function to send POST request and log the response
def send_post_request(data):
    response = requests.post(url, headers=headers, data=json.dumps(data))
    print(f"POST {url}")
    print(f"Input data: {json.dumps(data, indent=4)}")
    print(f"Status code: {response.status_code}")
    print(f"Result: {response.json()}")

# Sending the first POST request
send_post_request(data1)

# Sending the second POST request
send_post_request(data2)