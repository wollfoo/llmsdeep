import json
import sys

def validate_json(file_path):
    try:
        with open(file_path, 'r') as f:
            json.load(f)
        print(f"{file_path} is valid.")
    except json.JSONDecodeError as e:
        print(f"{file_path} is invalid: {e}")

if __name__ == "__main__":
    for file in sys.argv[1:]:
        validate_json(file)
