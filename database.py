import json

filename = 'db.json'
# entry = {'carl': 33}

def update(input_data):
    # 1. Read file contents
    with open(filename, "r") as file:
        data = json.load(file)

    # 2. Update json object
    data.update(input_data)

    # 3. Write json file
    with open(filename, "w") as file:
        json.dump(data, file)

def collections():
    # 1. Read file contents
    with open(filename, "r") as file:
        data = json.load(file)

    # 2. Update json object
    return data 

def get_collection(collection):
    # 1. Read file contents
    with open(filename, "r") as file:
        data = json.load(file)

    # 2. Get json object
    return data.get(collection)

def remove(collection):
    # 1. Read file contents
    with open(filename, "r") as file:
        data = json.load(file)

    # 2. Delete json object
    data.pop(collection)

    # 3. Write json file
    with open(filename, "w") as file:
        json.dump(data, file)
