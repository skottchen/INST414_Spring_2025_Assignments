import json

file_name = "artists_1_to_10.json"  # Replace with your actual file
with open(file_name, "r") as f:
    try:
        json_data = json.load(f)
        print("Valid JSON")
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
