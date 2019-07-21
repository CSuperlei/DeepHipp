import json


def dump_json(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(data)
    return data


if __name__ == '__main__':
    filename = './test_list.json'
    load_json(filename)