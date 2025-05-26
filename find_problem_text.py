import json

# Read the dataset with UTF-8 encoding
with open("src/llama_cookbook/datasets/alpaca_data.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Search for items with specific text patterns we identified in earlier analysis
for idx, item in enumerate(dataset):
    instruction = item.get("instruction", "")
    if "social network centered around" in instruction:
        print(f"Found item at index {idx}:")
        print(f"Instruction: {instruction}")
        break 