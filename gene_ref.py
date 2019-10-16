import json
import pdb

ref_json_path = "./data/unprocessed.test.json"
ref_text_path = "./test_ref.txt"

with open(ref_json_path) as fil:
	content = fil.read()

struct = json.loads(content)


with open(ref_text_path , "w") as fil:
	for x in struct:
		fil.write(x["abstract_og"] + "\n")

