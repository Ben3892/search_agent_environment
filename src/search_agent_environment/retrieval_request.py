import requests

# URL for your local FastAPI server
# url = "http://10.192.132.20:8000/retrieve_rerank"

import requests

url = "http://localhost:8000/search"
# 36937535
# Example payload
payload = {
    "queries": ["What is the capital of China?"],
    "topk": 5,
    "return_scores": True
}

# Send POST request
response = requests.post(url, json=payload)

# Raise an exception if the request failed
response.raise_for_status()

# Get the JSON response
retrieved_data = response.json()

print("Response from server:")
print(retrieved_data)
# search：用户输入一个query和topk，返回和这个query最相关的document的第一个passage
# visit：用户输入一个url、query，返回和这个url对应的document里面的passage里最相关的topk个passage
# import requests

# payload ={"items": 
#     [
#         {"url": "https://en.wikipedia.org/wiki/Politics_of_Berlin", "query": "What is the capital of German?"}
#     ]
# }
# resp = requests.post("http://localhost:8000/visit", json=payload)
# retrieved_data = resp.json()

# print("Response from server:")
# print(retrieved_data)