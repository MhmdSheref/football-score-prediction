import requests
import json

def fetch_openalex(query):
    url = f"https://api.openalex.org/works?search={query}&mailto=test@example.com"
    resp = requests.get(url)
    data = resp.json()
    for work in data.get('results', [])[:2]:
        print("TITLE:", work.get('title'))
        print("DOI:", work.get('doi'))
        print("AUTHORS:", [a['author']['display_name'] for a in work.get('authorships', [])])
        print("YEAR:", work.get('publication_year'))
        print("------")

print("--- Karlis Ntzoufras ---")
fetch_openalex("Karlis Ntzoufras bivariate Poisson")
print("--- Constantinou ---")
fetch_openalex("Constantinou pi-football bayesian networks")
print("--- Guyon FIFA ---")
fetch_openalex("Guyon Risk of Collusion FIFA")
print("--- Bivariate Poisson football margins ---")
fetch_openalex("Predicting football match results Karlis")
