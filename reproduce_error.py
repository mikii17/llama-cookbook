import json
import sys

# Reproduce the error by trying to force the decoding using charmap
# and specifically NOT using UTF-8 encoding
filename = "src/llama_cookbook/datasets/alpaca_data.json"

try:
    # This is what's likely happening in the code that's failing
    # It's trying to open the file without specifying an encoding, 
    # which would make Python use the system default (often 'charmap' on Windows)
    with open(filename, 'r', encoding='charmap') as f:
        # We'll add a breakpoint at exactly position 43444
        # to see what happens when we hit that character
        data = f.read(43443)  # Read just before the problematic byte
        print(f"Successfully read {len(data)} bytes")
        
        # Now read just the character at position 43444
        problematic_char = f.read(1)
        print(f"Character at position 43444: {repr(problematic_char)}")
        
        # Continue reading
        rest = f.read()
        print(f"Successfully read the rest of the file: {len(rest)} bytes")
except UnicodeDecodeError as e:
    print(f"Error: {e}")
    print(f"Error position: {e.start}")

# Let's see if we can fix it by explicitly specifying UTF-8 encoding
print("\nAttempting to fix by using UTF-8 encoding:")
try:
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print(f"Success! Loaded {len(data)} items using UTF-8 encoding")
except Exception as e:
    print(f"Error with UTF-8: {e}") 