import sys
import json
from src.llama_cookbook.datasets.alpaca_dataset import InstructionDataset

# Create a minimal dataset config class
class DummyConfig:
    data_path = "src/llama_cookbook/datasets/alpaca_data.json"

# Create a minimal tokenizer class
class DummyTokenizer:
    def encode(self, text):
        return []
    
    @property
    def eos_token_id(self):
        return 2

try:
    # Test loading the dataset with our modified code
    print("Testing fixed InstructionDataset class...")
    dataset = InstructionDataset(DummyConfig(), DummyTokenizer())
    print(f"Success! Dataset loaded with {len(dataset.ann)} items")
    
    # Print a sample of the data where we found problematic characters
    sample_idx = 0  # First sample
    for idx, item in enumerate(dataset.ann):
        if "posts" in item.get("instruction", "") and "social network" in item.get("instruction", ""):
            sample_idx = idx
            break
    
    print(f"\nSample item at index {sample_idx}:")
    instruction = dataset.ann[sample_idx].get("instruction", "")
    print(f"Instruction: {instruction[:200]}...")
    
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}") 