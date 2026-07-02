import requests
import json

def fetch_openalex(query):
    url = f"https://api.openalex.org/works?search={query}&mailto=test@example.com"
    resp = requests.get(url)
    data = resp.json()
    for work in data.get('results', [])[:3]:
        print("TITLE:", work.get('title'))
        print("DOI:", work.get('doi'))
        print("AUTHORS:", [a['author']['display_name'] for a in work.get('authorships', [])])
        print("YEAR:", work.get('publication_year'))
        print("------")

print("--- 48-team ---")
fetch_openalex("48-team FIFA World Cup")
print("--- Football Aleatoric ---")
fetch_openalex("football match prediction uncertainty")
print("--- Tertiary / Draw Prediction ---")
fetch_openalex("predicting football matches draw")
