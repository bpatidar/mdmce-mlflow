import requests
import json

import time as time
## Sample code that needs to be converted into java.
## Facility to mass upload item add data and fetch its standard value.

headers = {'Content-Type': 'application/json; format=pandas-split'}


input_data = {"pkey": ["ID003"],
              "description": ["Ear Curette McKesson Single-ended Handle with Grooves 4 mm Tip Oval Tip"]
              }

import pandas as pd

# Create DataFrame
df = pd.DataFrame(input_data)
df.to_csv("input_data_for_test.csv", index=False)
data = df.to_json(orient='split', index=False)

API_ENDPOINT = "http://127.0.0.1:5001/invocations"

start = time.time()
r = requests.post(API_ENDPOINT, data=data, headers=headers)  # , data = data)

end = time.time()
print("Result is :")
parsed = json.loads(r.text)
print(json.dumps(parsed, indent=4, sort_keys=True))
print("Time taken is", end - start)