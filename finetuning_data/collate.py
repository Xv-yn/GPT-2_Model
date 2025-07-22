import glob

# List your files in the correct order
files = [
    "alpaca_transformed_data.jsonl",
    "clean_medqa_test_0.jsonl",
    "clean_medqa_train_0.jsonl",
    "medical_qa_0.jsonl"
]

output_file = "combined_medical_data.jsonl"

with open(output_file, "w", encoding="utf-8") as outfile:
    for fname in files:
        with open(fname, "r", encoding="utf-8") as infile:
            for line in infile:
                outfile.write(line)