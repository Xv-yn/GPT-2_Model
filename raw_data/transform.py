import json

input_file = "medical_qa.jsonl"
output_file = "medical_qa_0.jsonl"

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        data = json.loads(line)
        new_data = {
            "input": data["instruction"],
            "output": data["response"]
        }
        json.dump(new_data, outfile)
        outfile.write("\n")
