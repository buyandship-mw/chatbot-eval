import json

def read_from_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def save_to_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)