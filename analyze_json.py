import json
import sys

def check_for_specific_byte(filename, byte_value=0x9d):
    print(f"Checking for byte 0x{byte_value:02x} in {filename}")
    
    # Read the file in binary mode
    with open(filename, 'rb') as f:
        data = f.read()
        
    # Look for the specific byte
    occurrences = []
    for i, b in enumerate(data):
        if b == byte_value:
            occurrences.append(i)
            if len(occurrences) >= 20:
                break
                
    print(f"Found {len(occurrences)} occurrences of byte 0x{byte_value:02x}")
    if occurrences:
        print(f"First few positions: {occurrences[:10]}")
        
        # Show context for the first few occurrences
        for pos in occurrences[:3]:
            start = max(0, pos - 40)
            end = min(len(data), pos + 40)
            print(f"\nContext around position {pos}:")
            print(f"Hex: {[hex(b) for b in data[pos-5:pos+5]]}")
            try:
                surrounding_text = data[start:pos].decode('utf-8', errors='replace') + "[0x9D]" + data[pos+1:end].decode('utf-8', errors='replace')
                print(f"Text: {surrounding_text}")
            except Exception as e:
                print(f"Error showing text: {e}")

def test_json_parsing_methods(filename):
    print("\nTesting different JSON parsing methods:")
    
    # Test 1: Using UTF-8 encoding
    print("\nTest 1: Using UTF-8 encoding with json.load()")
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"Success! Loaded {len(data)} items")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
    
    # Test 2: Using charmap encoding
    print("\nTest 2: Using charmap encoding with json.load()")
    try:
        with open(filename, 'r', encoding='charmap') as f:
            data = json.load(f)
            print(f"Success! Loaded {len(data)} items")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
    
    # Test 3: Read binary and decode with UTF-8
    print("\nTest 3: Read binary and decode with UTF-8")
    try:
        with open(filename, 'rb') as f:
            content = f.read().decode('utf-8')
            data = json.loads(content)
            print(f"Success! Loaded {len(data)} items")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
    
    # Test 4: Read binary and decode with charmap
    print("\nTest 4: Read binary and decode with charmap")
    try:
        with open(filename, 'rb') as f:
            content = f.read().decode('charmap')
            data = json.loads(content)
            print(f"Success! Loaded {len(data)} items")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    filename = "src/llama_cookbook/datasets/alpaca_data.json"
    check_for_specific_byte(filename, 0x9d)
    test_json_parsing_methods(filename) 