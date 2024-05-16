import json


def load_json(file):
    with open(file) as f:
        return json.load(f)
