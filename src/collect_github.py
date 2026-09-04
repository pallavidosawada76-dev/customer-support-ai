import requests

url = "https://api.github.com/repos/microsoft/vscode/issues"

params = {
    "state": "closed",
    "per_page": 10
}

response = requests.get(url, params=params)

print("Status:", response.status_code)

data = response.json()

print("Number of issues:", len(data))

for issue in data:
    print("\n==============================")
    print("Issue:", issue["number"])
    print("Title:", issue["title"])
    print("\nBody:")
    print(issue["body"])