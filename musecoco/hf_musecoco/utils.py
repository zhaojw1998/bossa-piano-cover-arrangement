import os
import json

jpath = os.path.join

def read_json(path):
    """
    Read a json file to a dict obj.
    """
    with open(path, "r", encoding="utf8") as f:
        data = f.read()
        data = json.loads(data)
    return data

def save_json(data, path, sort=False):
    with open(path, 'w', encoding='utf8') as f:
        f.write(json.dumps(data, indent=4, sort_keys=sort, ensure_ascii=False))

def create_dir_if_not_exist(dir):
    if not os.path.exists(dir):
        # os.mkdir(dir)
        os.makedirs(dir)