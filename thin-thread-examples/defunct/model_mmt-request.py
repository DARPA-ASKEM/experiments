import requests

res = requests.get('http://34.230.33.149:8771/api/biomodels/BIOMD0000000955')
model_mmt = res.json()
