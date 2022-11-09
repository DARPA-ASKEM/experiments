import requests

res = requests.get('http://34.230.33.149:8771/api/biomodels/BIOMD0000000955')

res = requests.post('http://34.230.33.149:8771/api/to_petrinet', json = res.json())
model_petri = res.json()
